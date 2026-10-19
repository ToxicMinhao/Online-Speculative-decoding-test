"""Immutable transfer certificates and version-correct causal reading (PDF pp4–7)."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import hashlib
import json
import time
import numpy as np
from .core import BankConfig, CorrectionBank, Factors, MetricMemory, fuse


def _hash(value):
    def default(x):
        if isinstance(x, np.ndarray):
            return {'shape': x.shape, 'dtype': str(x.dtype), 'sha256': hashlib.sha256(x.tobytes()).hexdigest()}
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=default, allow_nan=False).encode()).hexdigest()


def actions(allocated):
    occupied = [i for i, yes in enumerate(allocated) if yes]
    result = [np.zeros(len(allocated))]
    for i in occupied:
        a = np.zeros(len(allocated)); a[i] = 1.; result.append(a)
    for n, i in enumerate(occupied):
        for j in occupied[n+1:]:
            for weight in (.5, 1.):
                a = np.zeros(len(allocated)); a[i] = a[j] = weight; result.append(a)
    return result


def action_key(alpha): return ','.join(format(float(a), '.1f') for a in alpha)


@dataclass
class ReadResult:
    alpha: np.ndarray
    effective: Factors
    forecast: float
    diagnostics: dict


class BankState:
    """The evaluator is supplied only to adaptation; reads never recompute heads."""
    def __init__(self, dimension, *, config=None, cost_profile=None):
        self.core = CorrectionBank(dimension, config or BankConfig())
        self.memory = []
        self.protection = []
        self.cache = {}
        self.cost_profile = copy.deepcopy(cost_profile or {})
        self.write_log = []

    def snapshot(self):
        return copy.deepcopy(dict(schema='correction_bank_v02', core=self.core.state_dict(),
            memory=self.memory, protection=self.protection, cache=self.cache,
            cost_profile=self.cost_profile, write_log=self.write_log))

    @classmethod
    def from_snapshot(cls, state):
        if state['schema'] != 'correction_bank_v02': raise ValueError('Wrong bank version')
        obj = cls.__new__(cls)
        obj.core = CorrectionBank.from_state_dict(state['core'])
        for key in ('memory', 'protection', 'cache', 'cost_profile', 'write_log'):
            setattr(obj, key, copy.deepcopy(state[key]))
        return obj

    def identity(self): return _hash(self.snapshot())

    def _versions(self, alpha):
        return [int(self.core.versions[i]) if a else None for i, a in enumerate(alpha)]

    def _cost(self, context_length, active):
        bins = self.cost_profile.get('bins', {})
        if not bins:
            if self.memory: raise ValueError('Read with memory requires a qualified cycle-cost profile')
            return 1., 'cold_start_no_profile'
        edges = sorted(int(k) for k in bins)
        edge = next((e for e in edges if context_length <= e), edges[-1])
        entry = bins[str(edge)]
        # Unmeasured bins deliberately use the declared nearest available profile;
        # this is an estimate, never a measured deployment speedup.
        cost = float(entry['cycle_s'][str(active)])
        if not np.isfinite(cost) or cost <= 0: raise ValueError('Invalid cycle-cost estimate')
        return cost, str(edge)

    def read(self, descriptor, context_length):
        x = np.asarray(descriptor, dtype=np.float64)
        if x.ndim != 1 or not np.isfinite(x).all() or not np.isclose(np.linalg.norm(x), 1., atol=1e-5):
            raise ValueError('Read requires normalized prompt-only descriptor')
        options = actions(self.core.allocated)
        by_response = {}
        for block in self.memory: by_response.setdefault(block['response_id'], []).append(block)
        kernels = {rid: float(np.exp(-np.sum((x-np.asarray(bs[0]['descriptor']))**2)/2))
                   for rid, bs in by_response.items()}
        rows = []
        for alpha in options:
            key = action_key(alpha); weighted_gain = weighted_base = weighted_prefix_gain = 0.
            for rid, blocks in by_response.items():
                diffs=[]; frozen=[]; prefix_diffs=[]
                for block in blocks:
                    current = self.cache[block['block_id']][key]
                    zero = self.cache[block['block_id']][action_key(np.zeros(4))]
                    if current['versions'] != self._versions(alpha): raise ValueError('Stale routing evidence')
                    # For cap/EOS use measured committed progress, not 1+prefix.
                    diffs.append(current['progress']-zero['progress'])
                    frozen.append(zero['progress'])
                    prefix_diffs.append(current['prefix']-zero['prefix'])
                weighted_gain += kernels[rid]*float(np.mean(diffs))
                weighted_prefix_gain += kernels[rid]*float(np.mean(prefix_diffs))
                weighted_base += kernels[rid]*float(np.mean(frozen))
            ksum=sum(kernels.values())
            predicted_gain=weighted_gain/(1+ksum)
            base=weighted_base/ksum if ksum else 1.
            active=int(np.count_nonzero(alpha)); cost, cost_bin=self._cost(context_length,active)
            rows.append(dict(coefficients=alpha.tolist(), predicted_gain=predicted_gain,
                predicted_prefix_gain=weighted_prefix_gain/(1+ksum), frozen_progress=base,
                predicted_progress=base+predicted_gain, cycle_s=cost, cost_bin=cost_bin,
                utility=(base+predicted_gain)/cost, active_modules=active,
                active_rank=sum(self.core.modules[i].A.shape[0] for i,a in enumerate(alpha) if a)))
        best=max(r['utility'] for r in rows)
        tol=1e-12+1e-9*max(1.,abs(best))
        candidates=[(i,r) for i,r in enumerate(rows) if best-r['utility']<=tol]
        index,row=min(candidates,key=lambda ir:(ir[1]['active_rank'],ir[1]['active_modules'],ir[0]))
        if not self.memory: index,row=0,rows[0]
        alpha=options[index]
        return ReadResult(alpha, fuse(self.core.modules,alpha), row['predicted_gain'],
            dict(actions=rows, selected=row, memory_responses=len(by_response),
                 zero_gain_prior=1., kernel='exp(-squared_distance/2)',
                 forecast_scope='Local matching-block progress, not a guarantee of request speedup'))

    def refresh(self, evaluator, *, changed_owner=None):
        start=time.perf_counter(); calls=0
        for block in self.memory:
            values=self.cache.setdefault(block['block_id'],{})
            for alpha in actions(self.core.allocated):
                key=action_key(alpha)
                if key in values and values[key]['versions']==self._versions(alpha): continue
                result=evaluator(block, fuse(self.core.modules,alpha)); calls+=1
                values[key]=dict(versions=self._versions(alpha), prefix=int(result['prefix']),
                    progress=float(result['progress']))
        return dict(action_cache_refresh_s=time.perf_counter()-start, head_evaluations=calls,
                    changed_owner=changed_owner)

    def add_transfer(self, blocks, alpha, evaluator):
        """Insert actual served-action gains BEFORE owner trials; never source fits."""
        alpha=np.asarray(alpha,dtype=np.float64)
        if any(alpha[i] and not self.core.allocated[i] for i in range(4)):
            raise ValueError('Unused module in historical action')
        ids=[b['block_id'] for b in blocks]
        if len(ids)!=len(set(ids)) or set(ids)&set(b['block_id'] for b in self.memory):
            raise ValueError('Duplicate memory insertion')
        measured=[]
        for raw in blocks:
            block=copy.deepcopy(raw); block['coefficients']=alpha.copy()
            frozen=evaluator(block, fuse(self.core.modules,np.zeros(4)))
            served=evaluator(block, fuse(self.core.modules,alpha))
            gain=float(frozen['loss']-served['loss'])
            prefix_gain=int(served['prefix']-frozen['prefix'])
            record=dict(block_id=block['block_id'], response_id=block['response_id'],
                coefficients=alpha.copy(), frozen_loss=float(frozen['loss']),
                reference_loss=float(served['loss']), reference_prefix=int(served['prefix']),
                reference_margins=np.asarray(served['margins'],dtype=np.float64).copy(),
                kl_gain=gain, kl_protected=gain>1e-10, prefix_protected=prefix_gain>0,
                rho=.25, provenance='actual pre-write served action')
            block['reference']=copy.deepcopy(record)
            self.memory.append(block)
            if record['kl_protected'] or record['prefix_protected']: self.protection.append(record)
            measured.append(dict(block_id=block['block_id'], kl_gain=gain, prefix_gain=prefix_gain,
                progress_gain=float(served['progress']-frozen['progress']),
                kl_protected=record['kl_protected'],prefix_protected=record['prefix_protected']))
        return measured

    def protect(self, modules, evaluator):
        reasons=[]; details=[]; by_id={b['block_id']:b for b in self.memory}
        for record in self.protection:
            result=evaluator(by_id[record['block_id']],fuse(modules,record['coefficients']))
            if record['kl_protected']:
                bound=record['reference_loss']+.25*record['kl_gain']
                tol=1e-8+1e-7*max(1.,abs(bound),abs(result['loss']))
                if result['loss']>bound+tol: reasons.append('kl_protection')
            if record['prefix_protected']:
                prefix=record['reference_prefix']; refs=record['reference_margins'][:prefix]
                actual=np.asarray(result['margins'])[:prefix]
                positive=refs>0
                tol=1e-8+1e-7*np.maximum(1.,np.abs(refs))
                if np.any(actual[positive] < .75*refs[positive]-tol[positive]) or np.any(actual[positive]<=0):
                    reasons.append('prefix_protection')
                if result['prefix'] < prefix: reasons.append('native_precision')
            details.append(dict(block_id=record['block_id'], loss=float(result['loss']), prefix=int(result['prefix'])))
        return not reasons, dict(reasons=sorted(set(reasons)), records=details)

    def metric_memory(self):
        return [MetricMemory(response_id=b['response_id'],features=b['h'],weights=b['omega'],
                             coefficients=b['coefficients'],record_id=b['block_id']) for b in self.memory]

    def write(self, objective, evaluator):
        immutable=_hash(self.protection); started=time.perf_counter(); retention_s=0.
        def check(modules):
            nonlocal retention_s
            tick=time.perf_counter(); result=self.protect(modules,evaluator); retention_s+=time.perf_counter()-tick
            return result
        result=self.core.write(objective,memory=self.metric_memory(),protection_callback=check)
        if immutable!=_hash(self.protection): raise RuntimeError('Protection references mutated')
        valid, audit=check(self.core.modules)
        if not valid: raise RuntimeError('Committed bank violates cumulative protection')
        result['retention_s']=retention_s
        result['retention_audit']=audit
        result['cache_refresh']=self.refresh(evaluator,changed_owner=result.get('owner'))
        result['bank_write_wall_s']=time.perf_counter()-started
        # The complete trial/SVD receipt is returned and persisted separately by
        # training.py. Repeating every singular-value vector in every later
        # pre-read snapshot would make compact bank retention grow quadratically.
        self.write_log.append(copy.deepcopy({key: result.get(key) for key in (
            'owner', 'selected_owner', 'committed', 'accepted_steps', 'source_fit_gain',
            'allocated', 'versions', 'skip_reason', 'costs', 'retention_s',
            'cache_refresh', 'bank_write_wall_s')}))
        return result
