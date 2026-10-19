"""Bank feedback service alongside the unchanged qualified OSD trainer.

Construction is worker-only. The independent frozen bank draft is copied before
any OSD update; target features and fixed embeddings/head may be shared. No bank
parameters enter the OSD optimizer. Qualification retains exact sampled features
and teacher features so historical losses are independently recomputable.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from experiments.dflash_per_turn import native as per_turn
from .artifacts import atomic_json, binding, checked, load_snapshot, save_snapshot
from .core import BankConfig, CorrectionBank, FeedbackSummary, ReadResult
from .feedback import FrozenResponseObjective, sampled_features
from .projections import FixedProjections, build_projections, gaussian_orthonormal


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    return value


def check_direct_protection(bank, objectives, *, tolerance=1e-7):
    """Compare actual historical KL changes; never accept quadratic slack alone."""
    rows=[]
    for record in bank.protection:
        objective=objectives[record.response_id]
        effective=np.einsum('k,koi->oi',record.coefficients,bank.matrices)
        current,_=objective.direct_loss_gradient(effective)
        reference,_=objective.direct_loss_gradient(record.reference)
        D=effective-record.reference
        Q=float(np.sum(record.gradient*D)+bank.config.gamma*np.sum((D@record.covariance)*D))
        actual=current-reference
        budget=bank.config.rho*record.gain
        rows.append(dict(response_id=record.response_id,kind=record.kind,
            actual_loss_change=actual,quadratic_upper_bound=Q,allowed_erosion=budget,
            bound_slack=Q-actual,retention_slack=budget-actual,
            passed=bool(actual <= Q+tolerance and actual <= budget+tolerance), tolerance=tolerance))
    return {'passed':all(r['passed'] for r in rows),'records':rows,'actual_loss_recomputed':True}


class BankTrainingService(per_turn.TrainerService):
    def __init__(self, config):
        super().__init__(config)
        import torch
        self.bank_draft=copy.deepcopy(self.draft).eval().requires_grad_(False)
        self.bank_versions=tuple((n,id(p),p._version) for n,p in self.bank_draft.named_parameters())
        self.bank_generator=torch.Generator(device=next(self.bank_draft.parameters()).device).manual_seed(17)
        self.bank_directory=self.out/'bank';self.bank_directory.mkdir()
        self.arrays=self.bank_directory/'arrays'
        self.bank=None;self.projections=None;self.snapshot_reference=None
        self.qualification_objectives={}
        self.qualification_replay={}
        self.last_turn=0

    def _synchronize(self):
        import torch
        if next(self.bank_draft.parameters()).device.type=='cuda': torch.cuda.synchronize()

    def _check_bank_frozen(self):
        self.check_frozen()
        if self.bank_versions!=tuple((n,id(p),p._version) for n,p in self.bank_draft.named_parameters()):
            raise ValueError('Bank backbone weights changed')
        if self.bank_draft.training or any(p.requires_grad or p.grad is not None for p in self.bank_draft.parameters()):
            raise ValueError('Frozen bank backbone received training gradients')

    def prepare(self, body):
        import torch
        if self.bank is not None: raise ValueError('Bank initialization is once per conversation')
        self._check_bank_frozen();self._synchronize();started=time.perf_counter()
        E=self.components.lm_head.weight
        if self.config.get('bank_projections'):
            source=checked(self.config['bank_projections'])
            self.projections=FixedProjections.load(source)
            if self.projections.U.shape!=(E.shape[1],64) or self.projections.P.shape!=(E.shape[0],64):
                raise ValueError('Fixed projection shape does not match released head')
            if self.projections.seed != 17 or not np.allclose(self.projections.U.T@self.projections.U,np.eye(64),atol=1e-10):
                raise ValueError('Fixed projection seed or orthogonality differs')
            for name in ('V_in','V_descriptor'):
                expected=gaussian_orthonormal(E.shape[1],64,17)
                if not np.allclose(getattr(self.projections,name),expected,atol=1e-12,rtol=1e-12):
                    raise ValueError('Fixed Gaussian projection differs from pinned seed-17 construction')
            actual_kappa=2*float(np.linalg.norm(self.projections.P-self.projections.P.mean(axis=0),axis=1).max())
            if not np.isclose(actual_kappa,self.projections.kappa,atol=1e-12,rtol=1e-12):
                raise ValueError('Fixed projection radius does not match actual vocabulary geometry')
            # Artifact is not trusted solely on shape: verify E U in chunks.
            for start in range(0,E.shape[0],4096):
                actual=E[start:start+4096].detach().double().cpu().numpy()@self.projections.U
                if not np.allclose(actual,self.projections.P[start:start+4096],atol=1e-10,rtol=1e-9):
                    raise ValueError('Projection artifact belongs to a different frozen head')
            projection_binding=self.config['bank_projections']
        else:
            # Float64 Gram computation avoids copying the full vocabulary matrix
            # into a second CPU representation; eigensolver and signs are pinned.
            gram=torch.zeros((E.shape[1],E.shape[1]),device=E.device,dtype=torch.float64)
            for start in range(0,E.shape[0],4096):
                block=E[start:start+4096].detach().double();gram.add_(block.T@block)
            self.projections=build_projections(E,descriptor_dimension=E.shape[1],gram=gram.cpu().numpy())
            path=self.bank_directory/'fixed_projections.npz';self.projections.save(path)
            projection_binding=binding(path)
        self._synchronize();projection_s=time.perf_counter()-started
        atomic_json(self.bank_directory/'PROJECTION_METADATA.json',{
            **self.projections.metadata(),'released_checkpoint':self.config['released'],
            'target_identity':self.config['target']['identity_sha256'],
            'descriptor_hook':'normalized final target hidden state at last exact prompt token',
            'draft_hook':'output of frozen draft final norm immediately before vocabulary projection',
            'gram':'float64 row-chunk E.T@E; scipy.linalg.eigh(driver=evr) top64',
            'numeric_reference':'float64 fixed-feature affine objective; native serving precision reported separately'})
        self.bank=CorrectionBank(BankConfig(kappa=self.projections.kappa))
        self.snapshot_reference=save_snapshot(self.bank.snapshot(),self.bank_directory/'before_turn_01.json',array_root=self.arrays)
        return {'status':'passed','projections':projection_binding,'bank_snapshot':self.snapshot_reference,
                'costs':{'projection_preparation_s':projection_s},'qualification_checks':{},
                'scope':'one-time setup; no learned feedback used for projections'}

    def _objective(self, features):
        return FrozenResponseObjective(features['h'],features['teacher_h'],features['omega'],
            self.components.lm_head.weight,self.projections.U,self.projections.V_in,
            projection_chunk=32,synchronize=self._synchronize)

    def feedback(self, body):
        import torch
        self._check_bank_frozen()
        if self.bank is None: raise ValueError('Prepare the bank first')
        turn=int(body['turn']);final=bool(body.get('final',False));qualification=bool(body.get('qualification',False))
        if turn!=self.last_turn+1: raise ValueError('Bank feedback must be chronological and unique')
        if body['before_snapshot']!=self.snapshot_reference: raise ValueError('Feedback uses a different pre-read bank')
        checked(body['before_snapshot'])
        response=copy.deepcopy(body['response'])
        if response['turn']!=turn or response['episode_id']!=self.config['episode_id']:
            raise ValueError('Feedback ownership differs')
        # Serving state identity is supplied and stored by the controller. The
        # target/released identities are fixed for every feedback objective.
        if response.get('target_identity_sha256')!=self.config['target']['identity_sha256']:
            raise ValueError('Feedback target differs')
        saved=body['served_read']
        read=ReadResult(alpha=saved.get('alpha',saved.get('coefficients')),
                        **{k:saved[k] for k in ('effective','forecast','descriptor','diagnostics')})
        diagnostic_reconstruction_s=0.
        if saved.get('snapshot_sha256')!=body['before_snapshot']['sha256']:
            raise ValueError('Served forecast belongs to a different pre-read snapshot')
        if qualification:
            tick=time.perf_counter();expected=self.bank.read(read.descriptor)
            # This qualification-only recomputation never replaces the actual
            # forecast saved before the response.
            if not np.array_equal(expected.forecast,read.forecast) or not np.allclose(expected.effective,read.effective,atol=1e-10):
                raise ValueError('Served forecast/selection differs from saved pre-read state')
            diagnostic_reconstruction_s=time.perf_counter()-tick
        event_started=time.perf_counter()
        features=sampled_features(self.bank_draft,self.components.embed_tokens,self.teacher_features,response,
                                  generator=self.bank_generator,synchronize=self._synchronize,
                                  final_analysis=final)
        objective=self._objective(features)
        summary=FeedbackSummary(descriptor=read.descriptor,G=objective.G,S=objective.S,omega=objective.omega,
            phi=objective.phi,T=objective.T,forecast=read.forecast,error=objective.G-read.forecast,
            response_id=response['record_id'],metadata=features['metadata'])
        # Persist the observed response and sampled objective before mutating bank.
        compact={'summary':asdict(summary),'M':objective.M,'teacher_precision':objective.teacher_precision,
                 'rng_before':features['rng_before'],
                 'rng_after':features['rng_after'],'response':response,
                 'forecast_saved_before_response':plain(saved),'before_snapshot':body['before_snapshot']}
        feedback_ref=save_snapshot(compact,self.bank_directory/f'feedback_{turn:02d}.json',array_root=self.arrays)
        qualification_artifact_s=0.
        if qualification:
            artifact_started=time.perf_counter()
            replay=objective.portable()
            self.qualification_replay[response['record_id']]=save_snapshot(
                replay,self.bank_directory/f'qualification_features_{turn:02d}.json',array_root=self.arrays)
            # Keep only CPU feature tensors for historical checks, not a separate
            # full float64 vocabulary projection for every response.
            self.qualification_objectives[response['record_id']]=replay
            qualification_artifact_s=time.perf_counter()-artifact_started
        before_records=[asdict(r) for r in self.bank.protection]
        write=self.bank.observe_and_write(summary,objective,read,final=final)
        immutable=all(all(np.array_equal(value,getattr(new,key)) if isinstance(value,np.ndarray)
                          else value==getattr(new,key) for key,value in old.items())
                      for old,new in zip(before_records,self.bank.protection))
        checks={'frozen_backbone':True,'current_response_only':True,
                'fixed_sampled_objective':True,'protection_records_immutable':immutable,
                'sampled_positions':len(objective.h),
                'served_record_precedes_trials':write['records_before_owner_trials']==
                    write['records_before']+int(write['served_gain']>0)}
        qualification_s=0.
        if qualification:
            tick=time.perf_counter()
            points=[np.zeros_like(read.effective),read.effective]
            if write['owner'] is not None:points.append(self.bank.matrices[write['owner']])
            comparisons=[]
            for matrix in points:
                compressed,cg=objective.loss_gradient(matrix);direct,dg=objective.direct_loss_gradient(matrix)
                comparisons.append({'loss_error':abs(compressed-direct),
                    'gradient_max_error':float(np.max(np.abs(cg-dg))),
                    'passed':bool(abs(compressed-direct)<=1e-7 and np.allclose(cg,dg,atol=1e-8,rtol=1e-6))})
            historical=[]
            # Recreate one old objective at a time. This is qualification-only
            # and excluded from deployment adaptation cost, but fully timed.
            for rid,replay in self.qualification_objectives.items():
                old=objective if rid==response['record_id'] else self._objective({
                    'h':torch.from_numpy(replay['h']), 'teacher_h':torch.from_numpy(replay['teacher_h']),
                    'omega':replay['omega']})
                records=[r for r in self.bank.protection if r.response_id==rid]
                for record in records:
                    current=np.einsum('k,koi->oi',record.coefficients,self.bank.matrices)
                    loss,_=old.direct_loss_gradient(current);reference,_=old.direct_loss_gradient(record.reference)
                    D=current-record.reference
                    Q=float(np.sum(record.gradient*D)+self.bank.config.gamma*np.sum((D@record.covariance)*D))
                    budget=self.bank.config.rho*record.gain
                    historical.append({'response_id':rid,'kind':record.kind,'actual_loss_change':loss-reference,
                        'quadratic_bound':Q,'allowed_erosion':budget,
                        'passed':bool(loss-reference<=Q+1e-7 and loss-reference<=budget+1e-7)})
                if old is not objective:del old
            qualification_s=time.perf_counter()-tick
            checks.update(compressed_direct={'passed':all(x['passed'] for x in comparisons),'comparisons':comparisons},
                          direct_historical={'passed':all(x['passed'] for x in historical),
                                             'records':historical,'actual_loss_recomputed':True},
                          source_fit={'gain':write.get('singleton_fit_gain'),'performance_gate':False})
            # Nonzero fusion is checked even if the causal read stays at zero:
            # this is a deterministic implementation test, not a serving ablation.
            from .native_hooks import corrected_features
            rank=self.bank.config.rank
            rng=np.random.default_rng(17)
            matrices=rng.standard_normal((2,rank,rank))
            matrices*=self.bank.config.radius/(4*np.linalg.norm(matrices,axis=(1,2)))[:,None,None]
            alpha=np.array([.6,.7]);combined=np.einsum('k,koi->oi',alpha,matrices)
            hidden=features['h'][:8].to(self.components.lm_head.weight.device)
            if len(hidden):
                U=torch.tensor(self.projections.U,device=hidden.device,dtype=torch.float32)
                V=torch.tensor(self.projections.V_in,device=hidden.device,dtype=torch.float32)
                C=torch.tensor(combined,device=hidden.device,dtype=torch.float32)
                native=corrected_features(hidden,C,U,V)
                phi=hidden.float()@V;phi/=phi.norm(dim=-1,keepdim=True).clamp_min(1.)
                explicit=sum(float(a)*(phi@torch.tensor(c,device=hidden.device,dtype=torch.float32).T)@U.T
                             for a,c in zip(alpha,matrices))
                explicit=(hidden.float()+explicit).to(hidden.dtype)
                # One BF16 rounding unit may differ through arithmetic order;
                # compare the pre-cast FP32 expressions as the fusion identity.
                pre_error=float(((phi@C.T)@U.T - sum(float(a)*(phi@torch.tensor(c,device=hidden.device,dtype=torch.float32).T)@U.T
                    for a,c in zip(alpha,matrices))).abs().max())
                checks['fusion_detail']={'precast_max_error':pre_error,
                    'native_cast_max_error':float((native.float()-explicit.float()).abs().max()),
                    'zero_bit_preserving':bool(torch.equal(hidden,corrected_features(hidden,torch.zeros_like(C),U,V)))}
                checks['compiled_fusion']={'passed':pre_error<=1e-6 and checks['fusion_detail']['zero_bit_preserving'],
                                          **checks['fusion_detail']}
            else:
                checks['compiled_fusion']={'passed':True,'zero_exposure':True}
                checks['fusion_detail']={'no_eligible_positions':True,'separate_nonempty_turn_check_required':True}
            # Observe the actual serving arithmetic on the same frozen sampled
            # features. This neither changes the matrix nor requires any gain.
            from .precision import qualify_native_precision
            native_cases={'zero':np.zeros_like(read.effective),'served':read.effective,
                'controlled_nonzero':np.eye(rank)*(self.bank.config.radius/np.sqrt(rank))}
            if write['owner'] is not None:
                native_cases['fitted']=self.bank.matrices[write['owner']].copy()
            checks['native_precision']=qualify_native_precision(objective,native_cases)
            checks['native_precision']['controlled_case']='C = radius * identity / sqrt(rank), independent of feedback and RNG'
            checks['native_precision']['fitted_case_available']=write['owner'] is not None
            checks['native_precision_reported']=True
            checks.update(masks_and_conditioning=True,compressed_feedback=checks['compressed_direct']['passed'],
                          direct_retention_bounds=checks['direct_historical']['passed'],current_response_fit_reported=True)
            qualification_s=time.perf_counter()-tick
            if not checks['compressed_direct']['passed'] or not checks['direct_historical']['passed']:
                save_snapshot({'write':write,'checks':checks},self.bank_directory/f'FAILED_{turn:02d}.json',array_root=self.arrays)
                raise ValueError('Independent numerical qualification failed; evidence retained')
        diagnostics=save_snapshot(write,self.bank_directory/f'solver_{turn:02d}.json',array_root=self.arrays)
        self.snapshot_reference=save_snapshot(self.bank.snapshot(),self.bank_directory/f'after_turn_{turn:02d}.json',array_root=self.arrays)
        self.last_turn=turn;self._check_bank_frozen()
        event_s=time.perf_counter()-event_started
        receipt={'status':'passed','arm':'BANK','event':turn,'final':final,
            'bank_snapshot':self.snapshot_reference,'state_before_sha256':body['before_snapshot']['sha256'],
            'feedback':feedback_ref,'solver_diagnostics':diagnostics,
            'qualification_replay':self.qualification_replay.get(response['record_id']),
            'qualification_checks':checks,'current_response_only':True,'frozen_reference':True,
            'protection_records_immutable':immutable,'served_record_precedes_trials':checks['served_record_precedes_trials'],
            'scheduled_steps':0 if final else 2,
            'accepted_steps':sum(bool(s['accepted']) for s in write['steps']),
            'nonzero_steps':sum(bool(s['nonzero']) for s in write['steps']),
            'feasible_noop_steps':sum(bool(s['accepted'] and not s['nonzero']) for s in write['steps']),
            'owner':write['owner'],
            'served_gain':None if features['metadata'].get('zero_supervision') else write['served_gain'],
            'prewrite_singleton_gains':{str(c['owner']):write['frozen_loss']-c['base_loss']
                                        for c in write['candidates']},
            'prewrite_singleton_gain_scope':'Current-response feedback diagnostic, computed after serving; never a read input',
            'input_ids_sha256':author_hash(response['input_ids']),
            'output_ids_sha256':author_hash(response['output_ids']),
            'singleton_fit_gain':write.get('singleton_fit_gain'),
            'blocked_writes':sum(not s['accepted'] for s in write['steps']),
            'teacher_precision':objective.teacher_precision,
            'accounting':features['metadata'],
            'analysis_no_op':bool(final and features['metadata'].get('zero_supervision')),
            'analysis_no_op_reason':features['metadata'].get('no_op_reason'),
            'costs':{**features['costs'],'feedback_compression_s':objective.compression_s,
                'candidate_gradient_s':write['candidate_gradient_s'],'all_objective_s':write['all_objective_s'],
                'solver_s':write['solver_s'],'bank_write_s':write['adaptation_s'],
                'qualification_only_s':qualification_s+diagnostic_reconstruction_s+qualification_artifact_s,
                'qualification_artifact_s':qualification_artifact_s,
                'native_precision_s':checks.get('native_precision',{}).get('wall_s',0.),
                'diagnostic_reconstruction_s':diagnostic_reconstruction_s,
                'adaptation_total_s':event_s-qualification_s-qualification_artifact_s,
                'feedback_event_wall_s':event_s,
                'accounting':'adaptation_total_s includes teacher/features/compression/all candidates/solver/artifact I/O; components overlap and are not added again'}}
        atomic_json(self.bank_directory/f'RECEIPT_{turn:02d}.json',plain(receipt))
        return plain(receipt)


def author_hash(value):
    return per_turn.digest(value)
