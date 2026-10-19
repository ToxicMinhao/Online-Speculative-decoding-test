"""Causal bank feedback alongside the unchanged per-response author-path OSD."""
from __future__ import annotations
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from experiments.dflash_per_turn import native as per_turn
from experiments.dflash_correction_bank.artifacts import atomic_json, binding, checked, load_snapshot, save_snapshot
from experiments.dflash_correction_bank.projections import gaussian_orthonormal
from .core import BankConfig, Factors, fuse
from .state import BankState, _hash
from .feedback import FullHiddenObjective, sampled_features
from .costs import observe


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def qualify_native_fusion(objective):
    """Check the actual residual helper on sampled features on their head device.

    This local head diagnostic does not generate an answer or consume the bank
    sampler RNG. FP32 sum equivalence is checked; native rounding differences
    from association order are measured and do not require bitwise equality.
    """
    import torch
    from .native_hooks import corrected_features
    d=objective.width;k=min(32,d);rng=np.random.default_rng(17)
    modules=[Factors(rng.standard_normal((d,k))*.01/np.sqrt(k),
                     rng.standard_normal((k,d))/np.sqrt(d)) for _ in range(2)]
    pair=fuse(modules,[1.,1.])
    if not len(objective.features):
        return dict(passed=True,available=False,reason='no_eligible_sampled_features',
                    separate_nonempty_turn_check_required=True),pair
    hidden=torch.as_tensor(objective.features[:9],device=objective.device,dtype=objective.E_native.dtype)
    B=torch.tensor(pair.B,device=objective.device,dtype=torch.float32)
    A=torch.tensor(pair.A,device=objective.device,dtype=torch.float32)
    explicit=sum((hidden.float()@torch.tensor(m.A,device=objective.device).T)@
                 torch.tensor(m.B,device=objective.device).T for m in modules)
    compiled=(hidden.float()@A.T)@B.T
    corrected=corrected_features(hidden,B,A)
    explicit_native=(hidden.float()+explicit).to(hidden.dtype)
    zero=corrected_features(hidden,B[:,:0],A[:0])
    valid=bool(torch.allclose(compiled,explicit,atol=2e-5,rtol=2e-6) and torch.equal(zero,hidden))
    return dict(passed=valid,available=True,device=str(hidden.device),native_dtype=str(hidden.dtype),
        positions=len(hidden),module_rank=k,compiled_rank=pair.rank,
        pre_cast_max_error=float((compiled-explicit).abs().max()),
        native_cast_max_error=float((corrected.float()-explicit_native.float()).abs().max()),
        native_cast_changed_coordinates=int((corrected!=explicit_native).sum()),
        zero_bit_preserving=bool(torch.equal(zero,hidden)),atol=2e-5,rtol=2e-6,
        native_bitwise_association_equality_required=False,
        scope='Actual sampled features and native corrected_features helper; no response generation'),pair


def load_native_blocks(reference, response):
    """Join only exact captured native rows; no sampled replacement block."""
    if reference is None: raise ValueError('Canonical BANK response lacks frozen head capture')
    document=json.loads(checked(reference).read_text())
    if document['input_ids_sha256']!=per_turn.digest(response['input_ids']) or document['output_ids_sha256']!=per_turn.digest(response['output_ids']):
        raise ValueError('Head capture belongs to another canonical response')
    with np.load(checked(document['tensors']),allow_pickle=False) as data:
        hidden=data['frozen_h']
        blocks=[]
        for row in document['blocks']:
            eligible=np.asarray(row['eligible_mask'],dtype=bool)
            if not eligible.any():continue
            block=copy.deepcopy(row)
            block.update(h=hidden[row['hidden_index']][eligible].copy(),
                full_native_h=hidden[row['hidden_index']].copy(),
                native_row_indices=np.flatnonzero(eligible).tolist(),
                output_positions=[x for x,ok in zip(row['output_positions'],eligible) if ok],
                served_proposals=[x for x,ok in zip(row['proposal_token_ids'],eligible) if ok])
            blocks.append(block)
    return blocks,document


class BankTrainingService(per_turn.TrainerService):
    def __init__(self,config):
        super().__init__(config)
        import torch
        self.bank_draft=copy.deepcopy(self.draft).eval().requires_grad_(False)
        self.bank_versions=tuple((n,id(p),p._version) for n,p in self.bank_draft.named_parameters())
        self.bank_generator=torch.Generator(device=next(self.bank_draft.parameters()).device).manual_seed(17)
        self.bank_directory=self.out/'bank';self.bank_directory.mkdir()
        self.arrays=self.bank_directory/'arrays'
        self.bank=None;self.snapshot_reference=None;self.last_turn=0
        self.qualification_replay={};self._memory_objectives={}

    def _sync(self):
        import torch
        if next(self.bank_draft.parameters()).device.type=='cuda':torch.cuda.synchronize()

    def _frozen(self):
        self.check_frozen()
        if self.bank_versions!=tuple((n,id(p),p._version) for n,p in self.bank_draft.named_parameters()):
            raise ValueError('Frozen bank backbone changed')
        if self.bank_draft.training or any(p.requires_grad or p.grad is not None for p in self.bank_draft.parameters()):
            raise ValueError('Bank backbone entered training')

    def prepare(self,body):
        if self.bank is not None:raise ValueError('Prepare only once per stream')
        self._frozen();started=time.perf_counter();d=int(self.components.lm_head.weight.shape[1])
        projection=gaussian_orthonormal(d,64,17).astype(np.float32)
        path=self.bank_directory/'descriptor_projection.npz'
        np.savez(path,V_descriptor=projection)
        profile={}
        if self.config.get('qualified_cost_profile'):
            profile=json.loads(checked(self.config['qualified_cost_profile']).read_text())
        elif not self.config.get('bank_qualification'):
            raise ValueError('Main campaign requires qualification cost profile binding')
        self.bank=BankState(d,cost_profile=profile)
        atomic_json(self.bank_directory/'NUMERICAL_CONFIG.json',dict(core=asdict(self.bank.core.config),
            descriptor_seed=17,descriptor_hook='last prompt token final-normalized target feature',
            correction_hook='frozen DFlash final norm output before vocabulary projection',
            memory_dtype='FP32 full hidden h and mu',math_dtype='FP64',native_residual='FP32 then native BF16',
            qualification_loss_atol=2e-5,qualification_gradient_atol=2e-6,qualification_gradient_rtol=2e-4,
            retention_math_tolerance='1e-8+1e-7*max(1,abs(reference),abs(current))',
            memory_hash='sha256(seed17,stream,turn,output-relative block start)',
            native_output_ids_required=True,normalized_metric=True,
            absolute_kl_reported=False,head_vocab_chunk=4096,head_row_chunk=32,
            positive_kl_gain_threshold=1e-10,read_utility_tie_atol=1e-12,read_utility_tie_rtol=1e-9))
        self.snapshot_reference=save_snapshot(self.bank.snapshot(),self.bank_directory/'before_turn_01.json',array_root=self.arrays)
        return dict(status='passed',projections=binding(path),bank_snapshot=self.snapshot_reference,
            costs={'projection_preparation_s':time.perf_counter()-started},qualification_checks={})

    def _from_teacher(self,features):
        return FullHiddenObjective.from_teacher(features['h'],features['teacher_h'],features['omega'],
            self.components.lm_head.weight,synchronize=self._sync,
            native_h=features.get('native_h'),native_row_indices=features.get('native_row_indices'))

    def _evaluate(self,block,factors):
        key=block['block_id']
        if key not in self._memory_objectives:
            self._memory_objectives[key]=FullHiddenObjective(block['h'],block['mu'],block['omega'],
                self.components.lm_head.weight,synchronize=self._sync,
                native_h=block.get('native_h'),native_row_indices=block.get('native_row_indices'))
        obj=self._memory_objectives[key]
        native=obj.native_evaluate(factors,block['token_ids'],eos_token_ids=(128001,128009))
        prefix=int(native['prefix_lengths'][0])
        # Candidate rows omit the already committed target seed. A mismatch
        # produces a correction; a fully accepted interior block gets a bonus.
        # At EOS/cap clip to the remaining canonical output positions.
        remaining=int(block['remaining_output_tokens'])
        progress=min(remaining,1+prefix)
        if prefix and block['token_ids'][prefix-1] in (128001,128009):progress=prefix
        return dict(loss=obj.loss(factors),margins=obj.mathematical_margins(factors,block['token_ids']),
            native_margins=native['margins'],predictions=native['predictions'],prefix=prefix,progress=progress)

    def feedback(self,body):
        self._frozen();started=time.perf_counter()
        turn=int(body['turn']);final=bool(body['final']);qualification=bool(body['qualification'])
        response=body['response'];read=body['served_read']
        if turn!=self.last_turn+1:raise ValueError('Noncausal or duplicate feedback')
        if (response.get('turn')!=turn or response.get('episode_id')!=self.config['episode_id'] or
                response.get('target_identity_sha256')!=self.config['target']['identity_sha256']):
            raise ValueError('Feedback response/target ownership differs')
        if body['before_snapshot']!=self.snapshot_reference or read['snapshot_sha256']!=self.snapshot_reference['sha256']:
            raise ValueError('Feedback/read does not bind pre-request snapshot')
        checked(self.snapshot_reference)
        expected_read=self.bank.read(np.asarray(read['descriptor'],dtype=np.float64),len(response['input_ids']))
        if not np.array_equal(expected_read.alpha,np.asarray(read['coefficients'])):
            raise ValueError('Served action differs from its saved causal read state')
        alpha=np.asarray(read['coefficients']);served=fuse(self.bank.core.modules,alpha)
        before_records=_hash(self.bank.protection);old_records=copy.deepcopy(self.bank.protection)
        if not final and body.get('cost_profile') and not self.bank.cost_profile:
            self.bank.cost_profile=copy.deepcopy(body['cost_profile'])
        blocks,capture=load_native_blocks(body['native_head_capture'],response)
        features=sampled_features(self.bank_draft,self.components.embed_tokens,self.teacher_features,response,
            generator=self.bank_generator,native_blocks=blocks,synchronize=self._sync,final_analysis=final)
        objective=self._from_teacher(features)
        memory=[];precision=[];canonical_checks=[];compression_s=objective.compression['compression_s']
        for block in features['memory_blocks']:
            mo=self._from_teacher(block);compression_s+=mo.compression['compression_s']
            stable=dict(block_id=f"{response['record_id']}:block{block['block_start']}",
                response_id=response['record_id'],h=mo.features,mu=mo.mu,omega=mo.weights,
                native_h=mo.native_h,native_row_indices=mo.native_row_indices,
                token_ids=block['token_ids'],output_positions=block['output_positions'],
                block_start=block['block_start'],descriptor=np.asarray(read['descriptor'],dtype=np.float64),
                remaining_output_tokens=len(response['output_ids'])-min(block['output_positions']),
                metadata=block['metadata'],teacher_compression=mo.compression,
                precision='FP32 stored original native BF16 h; FP32 full teacher moment')
            self._memory_objectives[stable['block_id']]=mo
            # Independent native serving proposals validate head reconstruction.
            check=self._evaluate(stable,served)
            expected=block['metadata']['served_proposals']
            canonical_checks.append(dict(block_id=stable['block_id'],passed=check['predictions'].tolist()==expected,
                native_reconstructed=check['predictions'].tolist(),actual_served=expected))
            if qualification:
                self.qualification_replay[stable['block_id']]=mo.portable()
            memory.append(stable)
        if not all(c['passed'] for c in canonical_checks):
            save_snapshot({'checks':canonical_checks,'memory':memory},self.bank_directory/f'FAILED_NATIVE_{turn:02d}.json',array_root=self.arrays)
            raise ValueError('Stored native features do not reconstruct actual served proposals')
        # The selected source sample is fixed before any owner trial. Arrays are
        # not retained for the whole campaign; qualification keeps exact replay.
        persistence_started=time.perf_counter()
        feedback_ref=save_snapshot(dict(response=response,read=read,metadata=features['metadata'],
            reservation=features['reservation'],rng_before=features['rng_before'],rng_after=features['rng_after'],
            memory=memory,canonical_checks=canonical_checks,teacher_compression=objective.compression),
            self.bank_directory/f'feedback_{turn:02d}.json',array_root=self.arrays)
        qualification_replay=None
        if qualification:
            qualification_replay=save_snapshot(dict(source=objective.portable(),memory=self.qualification_replay),
                self.bank_directory/f'qualification_features_{turn:02d}.json',array_root=self.arrays)
        persistence_s=time.perf_counter()-persistence_started
        protection_start=time.perf_counter()
        if final:
            temporary=BankState.from_snapshot(self.bank.snapshot())
            gains=temporary.add_transfer(memory,alpha,self._evaluate)
            write=dict(owner=None,accepted_steps=0,committed=False,trials=[],source_fit_gain=None,costs={})
        else:
            # Mandatory order: served transfer records first, independently
            # trial all owners next. Post-write source fits create no certificate.
            gains=self.bank.add_transfer(memory,alpha,self._evaluate)
            if len(objective.features):
                write=self.bank.write(objective,self._evaluate)
            else:
                write=dict(owner=None,accepted_steps=0,committed=False,trials=[],source_fit_gain=None,
                    skip_reason=features['metadata'].get('no_op_reason','sampled_objective_has_no_eligible_positions'),costs={})
                self.bank.write_log.append(copy.deepcopy(write))
            observe(self.bank.cost_profile,body.get('cycle_observations',[]),int(np.count_nonzero(alpha)))
        protection_and_write_s=time.perf_counter()-protection_start
        if old_records!=[] and before_records!=_hash(self.bank.protection[:len(old_records)]):
            raise ValueError('Historical references or budgets changed')
        actual_positions=set(features['metadata']['unique_supervised_position_ids'])
        response_start=len(response['input_ids']);response_end=response_start+len(response['output_ids'])
        mask_valid=(all(response_start<=pos<response_end for pos in actual_positions) and
                    not actual_positions.intersection(features['metadata'].get('excluded_memory_absolute_positions',[])))
        if not mask_valid:raise ValueError('Newest-response or reserved-memory supervision mask failed')
        checks=dict(frozen_backbone=True,current_response_only=mask_valid,fixed_sampled_objective=True,
            protection_records_immutable=True,served_record_precedes_trials=True,
            masks_and_conditioning=mask_valid and all(c['passed'] for c in canonical_checks),
            rank_preserving_updates=all(m.rank<=32 for m in self.bank.core.modules),
            sampled_positions=len(objective.features))
        owner=write.get('owner');fitted=self.bank.core.modules[owner] if owner is not None else served
        # Native precision/source-fitting is reported without making a win a gate.
        source_fit_started=time.perf_counter()
        source_fit=dict(frozen_loss=0.,served_loss_change=objective.loss(served),
            fitted_singleton_loss_change=objective.loss(fitted) if owner is not None else None,
            diagnostic_correction_loss_change=objective.loss(fitted),fitted_owner=owner,
            committed_write=bool(write.get('committed')),
            diagnostic_correction_kind='fitted_singleton' if owner is not None else 'unchanged_served_action',
            source_response_only=True)
        labels=np.asarray(features['metadata'].get('label_positions',[]),dtype=np.int64)
        weights=np.asarray(features['metadata'].get('position_weights',[]),dtype=np.float64)
        source_positions=labels[weights>0].tolist()
        source_tokens=[response['output_ids'][pos-len(response['input_ids'])] for pos in source_positions]
        if len(source_tokens)!=len(objective.features):raise ValueError('Source-fit diagnostic labels are not aligned')
        source_fit['native_correct_positions']={}
        for name,factors in (('frozen',Factors.zero(self.bank.core.d)),('served',served),('fitted',fitted)):
            native=objective.native_evaluate(factors,source_tokens)
            source_fit['native_correct_positions'][name]=int(native['correct'].sum())
        source_fit.update(native_position_presentations=len(source_tokens),
            unique_output_positions=len(set(source_positions)),
            native_scope='Same-response fit at sampled eligible positions; repeated anchor positions retained. '
                         'These noncontiguous rows are not a matching-prefix trajectory or future transfer.')
        source_fit_diagnostic_only_s=time.perf_counter()-source_fit_started
        qualification_s=0.
        if qualification:
            tick=time.perf_counter();comparisons=[]
            controlled=Factors(np.eye(self.bank.core.d,1,dtype=np.float32)*.01,
                               np.eye(1,self.bank.core.d,dtype=np.float32))
            cases={'zero':Factors.zero(self.bank.core.d),'served':served,'fitted':fitted,'controlled':controlled}
            actual_fusion,controlled_pair=qualify_native_fusion(objective)
            cases['controlled_pair']=controlled_pair
            for name,factors in cases.items():
                value,gradient=objective.loss_gradient(factors);direct,dg=objective.direct_loss_gradient(factors)
                error=abs(value-direct);gerr=float(np.max(np.abs(gradient.dense()-dg)))
                comparisons.append(dict(case=name,loss_error=error,gradient_max_error=gerr,
                    passed=bool(error<=2e-5*max(1.,abs(direct)) and np.allclose(gradient.dense(),dg,atol=2e-6,rtol=2e-4))))
                precision.append(dict(case=name,**objective.precision_diagnostic(factors)))
            historical=[]
            for record in self.bank.protection:
                replay=self.qualification_replay[record['block_id']]
                historical_obj=FullHiddenObjective(replay['h'],replay['mu'],replay['omega'],
                    self.components.lm_head.weight,teacher_h=replay['teacher_h'],synchronize=self._sync,
                    native_h=replay.get('native_h'),native_row_indices=replay.get('native_row_indices'))
                actual,_=historical_obj.direct_loss_gradient(fuse(self.bank.core.modules,record['coefficients']))
                # Direct and stored-moment losses share zero reference; disclose
                # their discrepancy separately instead of calling storage exact.
                compressed=historical_obj.loss(fuse(self.bank.core.modules,record['coefficients']))
                bound=record['reference_loss']+.25*record['kl_gain']
                historical.append(dict(block_id=record['block_id'],direct_loss_change=actual,
                    compressed_loss_change=compressed,storage_discrepancy=actual-compressed,
                    bound=bound,kl_protected=record['kl_protected'],
                    passed=bool((not record['kl_protected']) or actual<=bound+2e-5)))
            from .qualification import fusion_fixture
            checks.update(compiled_fusion=fusion_fixture(),
                sampled_native_fusion=actual_fusion,
                compressed_direct={'passed':all(c['passed'] for c in comparisons),'comparisons':comparisons},
                direct_historical={'passed':all(c['passed'] for c in historical),'records':historical},
                source_fit=source_fit,native_precision_reported=True,
                native_precision={'status':'measured','cases':precision})
            qualification_s=time.perf_counter()-tick
            if (not checks['compressed_direct']['passed'] or not checks['direct_historical']['passed'] or
                    not actual_fusion['passed']):
                save_snapshot(checks,self.bank_directory/f'FAILED_NUMERIC_{turn:02d}.json',array_root=self.arrays)
                raise ValueError('Direct compressed/native qualification failed')
        # Retain compact memory, references and all trials, but no unique source
        # capture is deleted before its selected records and metadata are durable.
        persist=time.perf_counter()
        diagnostics=save_snapshot(write,self.bank_directory/f'solver_{turn:02d}.json',array_root=self.arrays)
        self.snapshot_reference=save_snapshot(self.bank.snapshot(),self.bank_directory/f'after_turn_{turn:02d}.json',array_root=self.arrays)
        persistence_s+=time.perf_counter()-persist
        self.last_turn=turn;self._frozen()
        reasons=[reason for trial in write.get('trials',[]) for step in trial.get('attempts',[]) for reason in step.get('reasons',[])]
        result=dict(status='passed',arm='BANK',event=turn,final=final,
            bank_snapshot=self.snapshot_reference,state_before_sha256=body['before_snapshot']['sha256'],
            feedback=feedback_ref,solver_diagnostics=diagnostics,qualification_replay=qualification_replay,
            qualification_checks=checks,current_response_only=True,frozen_reference=True,
            protection_records_immutable=True,served_record_precedes_trials=True,
            scheduled_steps=0 if final else 2,accepted_steps=write['accepted_steps'],owner=owner,
            skip_reason=write.get('skip_reason'),
            source_fit=source_fit,singleton_fit_gain=write.get('source_fit_gain'),
            blocked_writes=int(not final and not write.get('committed')),write_rejection_reasons=reasons,
            protection_failures=[r for r in reasons if r in ('kl_protection','prefix_protection','native_precision')],
            new_memory_blocks=0 if final else len(memory),no_new_routing_evidence=not memory,
            realized_progress_gain=float(np.mean([r['progress_gain'] for r in gains])) if gains else None,
            prewrite_transfer=gains,accounting=features['metadata'],
            input_ids_sha256=per_turn.digest(response['input_ids']),output_ids_sha256=per_turn.digest(response['output_ids']),
            costs={**features['costs'],'feedback_compression_s':compression_s,
                'capture_costs_already_in_canonical_request':capture['costs'],
                'protection_and_write_s':protection_and_write_s,**write.get('costs',{}),
                'retention_s':write.get('retention_s',0.),
                'action_cache_refresh_s':write.get('cache_refresh',{}).get('action_cache_refresh_s',0.),
                'checkpoint_persistence_s':persistence_s,'qualification_only_s':qualification_s,
                'source_fit_diagnostic_only_s':source_fit_diagnostic_only_s,
                'feedback_event_wall_s':time.perf_counter()-started,
                'accounting':'Subtimers explain outer cost; never sum overlapping timers or add canonical capture twice.'})
        atomic_json(self.bank_directory/f'RECEIPT_{turn:02d}.json',plain(result))
        return plain(result)
