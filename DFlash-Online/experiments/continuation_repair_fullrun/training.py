"""Fresh BASE/TAIL trajectories on the unchanged native event-training loop.

This wrapper owns identities, complete recovery checks and honest component
logging. Sampling, model arithmetic, accumulation, clipping and Adam stay native.
"""
from copy import deepcopy
import math
from pathlib import Path
import time

import torch

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences, _finite
from experiments.opencode_native_comparison.training import (
    recovery_size_plan, validate_recovery_file, initial_witness as native_initial_witness,
)
from experiments.persistent_prefix_withholding.qualification import isolated_backend_for_load
from experiments.unchanged_c_mathqa_transfer.state import validate_allocation_observation
from online_feedback.kplus import install_kplus, KPlusRedistribution
from online_feedback.prefix_allocation import install_prefix_allocation, prefix_allocation_contract
from online_feedback.repair_conditional_kl import install_repair_conditional_kl
from online_feedback.continuation_repair import install_continuation_repair, ContinuationRepair
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer, SCHEMA as STATE_SCHEMA

from .configuration import validate_config_text
from .contract import experiment_contract, ARMS, EVENTS, WINDOW, PASSES

CONTRACT_KEY = 'continuation_repair_fullrun_v1'
WITNESS_SCHEMA = 'continuation_repair_fullrun_initial_v1'
QUALIFICATION_SCHEMA = 'continuation_repair_fullrun_disposable_v1'


def component_logging(observation):
    """Expose effective FKL/PAIR components without touching autograd or metrics."""
    sums, ratios = observation['sum_metrics'], observation['ratio_metrics']
    fkl, denominator = ratios['forward_kl']
    pair, pair_denominator = ratios['repair_pair_auxiliary']
    actual, objective_denominator = ratios['objective_loss']
    if denominator != pair_denominator or denominator != objective_denominator:
        raise ValueError('Components must retain the one original response denominator')
    fk_correction = sums.get('continuation_fk_correction', 0.)
    pair_correction = sums.get('continuation_pair_correction', 0.)
    effective_fkl, effective_pair = fkl+fk_correction, pair+pair_correction
    # Reordered scalar logging can round differently from the original FP32 sum.
    tolerance = 2e-7 + 5e-6 * max(abs(actual), abs(effective_fkl)+abs(effective_pair))
    if not all(math.isfinite(float(x)) for x in (actual,effective_fkl,effective_pair,denominator)):
        raise ValueError('Nonfinite effective component logging')
    if abs(effective_fkl+effective_pair-actual) > tolerance:
        raise ValueError('Effective FKL and PAIR do not reconcile with the actual loss')
    return dict(native_C_K_FKL_numerator=fkl,native_PAIR_numerator=pair,
        continuation_fk_correction=fk_correction,continuation_pair_correction=pair_correction,
        effective_FKL_numerator=effective_fkl,effective_PAIR_numerator=effective_pair,
        effective_total_numerator=actual,denominator=denominator,
        effective_FKL=effective_fkl/denominator,effective_PAIR=effective_pair/denominator,
        effective_total=actual/denominator,reconciliation_error=effective_fkl+effective_pair-actual,
        reconciliation_tolerance=tolerance,scope='one original response; native old metric fields remain base components')


def validate_event_receipt(event, *, arm, binding, event_number, require_completed=True, trainer_status=None):
    if arm not in ARMS or not 1 <= event_number <= EVENTS:
        raise ValueError('Require this own BASE/TAIL event1..58')
    expected = dict(method=arm,dataset='OpenCodeInstruct',event_id=f'event_{event_number:03d}',
        objective='osd_first_error',passes=2,response_presentations=256,optimizer_steps=128,
        start_step=(event_number-1)*128,end_step=event_number*128,reduction='response_mean')
    if any(event.get(k)!=v for k,v in expected.items()) or event.get(CONTRACT_KEY)!=experiment_contract(arm,binding):
        raise ValueError('Own full-run event contract or accounting changed')
    if require_completed and event.get('status')!='completed':
        raise ValueError('Own event is incomplete')
    ids=[x['record_id'] for x in event['features']]
    rows=event['records']
    if len(ids)!=128 or len(set(ids))!=128 or len(rows)!=256:
        raise ValueError('Event must contain128 unique responses and256 presentations')
    for ordinal,row in enumerate(rows):
        stepped=bool(ordinal%2)
        if (row['pass']!=ordinal//128 or row['optimizer_stepped'] is not stepped or
            row['optimizer_step_after']!=(event_number-1)*128+(ordinal+1)//2 or
            row['learning_rate']!=(3e-4 if stepped else None)):
            raise ValueError('Native order, accumulation or optimizer counter changed')
        draws=row['sampled_anchors']
        if len(draws)!=1:
            raise ValueError('Expected one natural anchor draw per presentation')
        positions,keep=draws[0]['positions'],draws[0]['keep_mask']
        if (len(positions)!=1 or len(keep)!=1 or not 0<len(positions[0])<=512 or
            len(positions[0])!=len(keep[0]) or any(type(p) is not int or p<0 for p in positions[0]) or
            any(type(k) is not bool for k in keep[0])):
            raise ValueError('Malformed native anchor draw')
        observation=row['response_observation']
        if (observation['record_id']!=row['record_id'] or observation['pass_index']!=row['pass'] or
            observation['sampled_anchors']!=draws):
            raise ValueError('Original forward observation lost its identity')
        validate_allocation_observation(observation)
        if observation['effective_components']!=component_logging(observation):
            raise ValueError('Effective component receipt changed')
        values=[observation['loss'],*observation['sum_metrics'].values(),*row['metrics'].values()]
        values.extend(v for pair in observation['ratio_metrics'].values() for v in pair)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Nonfinite native original-response observation')
        if ('continuation_repair' in observation)!=(arm=='TAIL'):
            raise ValueError('TAIL observation flags differ from the installed algorithm')
    for offset in (0,128):
        if sorted(x['record_id'] for x in rows[offset:offset+128])!=sorted(ids):
            raise ValueError('Each shuffled pass must present each own response once')
    if trainer_status is not None:
        wanted=dict(global_step=event_number*128,presentations=event_number*256,
            completed_events=[f'event_{n:03d}' for n in range(1,event_number+1)],
            accumulation_remainder=0,objective='osd_first_error',horizon=7424)
        if any(trainer_status.get(k)!=v for k,v in wanted.items()):
            raise ValueError('Live trainer is not at the complete event boundary')
    return dict(status='passed',arm=arm,completed_event=event_number,**expected)


def validate_snapshot(state, *, arm, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe, expected_cuda_rng_states=1):
    """Check the actual BASE/TAIL state, without relabelling it as an older arm."""
    import random
    import numpy as np
    from experiments.osd_components.checkpoints import EXPECTED_KEYS
    validate_config_text(config_text,original_config_text)
    required={'event_schema','draft_state_dict','backend','rng','global_step','presentations',
        'events','strategy','run_id','world_size','config_text','algorithm_contract',
        'reduction','accumulation_steps','horizon'}
    if set(state)!=required or type(completed_event) is not int or not 0<=completed_event<=58:
        raise ValueError('Malformed complete full-run state')
    if (expected_contract.get(CONTRACT_KEY)!=experiment_contract(arm,binding) or
        differences(state['algorithm_contract'],expected_contract) or
        expected_contract['online_target_objective']['name']!='osd_first_error'):
        raise ValueError('Foreign or changed BASE/TAIL numerical contract')
    expected=dict(event_schema=STATE_SCHEMA,strategy='dflash',world_size=1,config_text=config_text,
        reduction='response_mean',accumulation_steps=2,horizon=7424,
        global_step=completed_event*128,presentations=completed_event*256)
    if any(state.get(k)!=v for k,v in expected.items()) or len(state['events'])!=completed_event:
        raise ValueError('Complete event counters/configuration differ')
    seen=set()
    for index,event in enumerate(state['events'],1):
        validate_event_receipt(event,arm=arm,binding=binding,event_number=index,require_completed=False)
        ids={x['record_id'] for x in event['features']}
        if seen&ids:raise ValueError('An own-feedback request was reused across events')
        seen.update(ids)
    weights=state['draft_state_dict']
    if set(weights)!=EXPECTED_KEYS or set(state['backend'])!={'optimizer'}:
        raise ValueError('Unexpected trainable tensor/backend membership')
    for name,value in weights.items():
        if not torch.is_tensor(value) or value.dtype!=torch.bfloat16 or not value.numel():
            raise ValueError('Invalid native draft tensor: '+name)
        _finite(value,name)
    optimizer=state['backend']['optimizer'];recipe=optimizer['recipe'];saved=optimizer['optimizer_state_dict']
    if differences(recipe,expected_recipe) or recipe['type']!='direct_bf16_adamw' or recipe['version']!=1:
        raise ValueError('Native optimizer recipe changed')
    if len(saved['param_groups'])!=1:raise ValueError('Expected one Adam group')
    group=saved['param_groups'][0];options={k:v for k,v in group.items() if k!='params'}
    if (differences(options,recipe['optimizer_options']) or recipe['max_grad_norm']!=1. or
        options['lr']!=3e-4 or options['weight_decay']!=.01 or options['betas']!=(.9,.999) or
        options['eps']!=1e-8 or options['amsgrad'] is not False):
        raise ValueError('Qualified Adam settings changed')
    ids=group['params'];parameters=recipe['parameters']
    if (len(ids)!=58 or len(parameters)!=58 or len(set(ids))!=58 or
        {p[0] for p in parameters}!=set(weights) or set(saved['state'])!=(set(ids) if completed_event else set())):
        raise ValueError('Native Adam parameters/moments missing')
    for (name,shape,dtype),identifier in zip(parameters,ids):
        weight=weights[name]
        if tuple(shape)!=tuple(weight.shape) or dtype!=str(weight.dtype):raise ValueError('Adam tensor binding changed')
        if not completed_event:continue
        moment=saved['state'][identifier]
        if set(moment)!={'step','exp_avg','exp_avg_sq'} or moment['step'].numel()!=1 or moment['step'].item()!=state['global_step']:
            raise ValueError('Native Adam counter changed')
        for key in ('exp_avg','exp_avg_sq'):
            if moment[key].dtype!=torch.bfloat16 or moment[key].shape!=weight.shape:raise ValueError('Adam moment precision changed')
            _finite(moment[key],name+'.'+key)
    rng=state['rng']
    if (set(rng)!={'python','numpy','torch','cuda'} or len(rng['cuda'])!=expected_cuda_rng_states):
        raise ValueError('Complete RNG state missing')
    for value in [rng['torch'],*rng['cuda']]:
        if value.dtype!=torch.uint8 or value.ndim!=1 or not value.numel():raise ValueError('Malformed RNG tensor')
    random.Random(0).setstate(rng['python']);np.random.RandomState(0).set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch'])
    return dict(status='passed',arm=arm,completed_event=completed_event,global_step=state['global_step'],
        presentations=state['presentations'],adam_state_entries=len(saved['state']),actual_arm_validated=True)


def initial_witness(state):
    result=native_initial_witness(state);result['schema']=WITNESS_SCHEMA
    return result


def compare_initial_witnesses(first,second):
    import yaml
    views=[];bindings=[]
    for witness in (first,second):
        if (witness['schema']!=WITNESS_SCHEMA or witness['global_step'] or witness['presentations'] or
            witness['events'] or witness['backend']['optimizer']['optimizer_state_dict']['state']):
            raise ValueError('Require fresh complete released initializations')
        contract=deepcopy(witness['algorithm_contract']);own=contract.pop(CONTRACT_KEY)
        if own!=experiment_contract(own['arm'],own['binding']):raise ValueError('Changed initial method contract')
        bindings.append(own['binding'])
        config=yaml.safe_load(witness['config_text'])
        for key in ('run_id','output_dir'):config.pop(key,None)
        views.append(dict(weights=witness['weights'],backend=witness['backend'],rng=witness['rng'],
            config=config,contract=contract,reduction=witness['reduction'],
            accumulation_steps=witness['accumulation_steps'],horizon=witness['horizon']))
    if bindings[0]!=bindings[1] or differences(*views):raise ValueError('Independent BASE/TAIL initial scientific state differs')
    return dict(status='passed',exact_weights_empty_Adam_RNG=True,rng_overwritten=False,
        allowed_changes=['run/output identity','arm and declared continuation flag'])


class FullContinuationTrainer(native.FullKTrainer):
    def __init__(self,config_path,output_root,*,arm,binding,original_config_text,
                 resume_state=None,completed_event=0,expected_cuda_rng_states=1,
                 qualification_only=False,adam_ledger=None):
        experiment_contract(arm,binding)
        if type(completed_event) is not int or not 0<=completed_event<=58 or ((completed_event>0)!=(resume_state is not None)):
            raise ValueError('Resume requires this arm\'s exact complete boundary')
        self.arm,self.binding=arm,binding
        self.original_config_text=original_config_text
        self.completed_event,self.expected_cuda_rng_states=completed_event,expected_cuda_rng_states
        self.qualification_only,self.adam_ledger=qualification_only,adam_ledger
        self.response_records,self.window_records,self.work,self._work_handles=[],[],{},[]
        self.step_observer=None
        SpecForgeEventTrainer.__init__(self,config_path,output_root,resume_state=resume_state,
            objective='osd_first_error',reduction='response_mean',ranking_lambda=0.)

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        validate_config_text(self.config_text,self.original_config_text)

    def _build_runtime(self):
        import rich.style
        SpecForgeEventTrainer._build_runtime(self)
        model=self.bundle.model
        install_kplus(model);install_prefix_allocation(model,relative_cap=.25)
        install_repair_conditional_kl(model,enabled=True)
        install_continuation_repair(model,enabled=self.arm=='TAIL')
        self.contract[CONTRACT_KEY]=experiment_contract(self.arm,self.binding)
        self._install_observations()
        ordinary_forward,ordinary_step=self._native_forward,self._native_step
        def forward(*args,**kwargs):
            self.check_installed_arm()
            output=ordinary_forward(*args,**kwargs)
            if not bool(torch.isfinite(output.loss).all()):raise ValueError('Nonfinite full-run loss')
            return output
        def step():
            if self.step_observer is not None:
                if not self.qualification_only:raise ValueError('Gradient observer is qualification-only')
                self.step_observer(self)
            token=self.adam_ledger.reserve(self.arm) if self.adam_ledger is not None else None
            result=ordinary_step()
            if self.adam_ledger is not None:self.adam_ledger.complete(token)
            return result
        self._native_forward,self._native_step=forward,step
        ordinary_backward=self.backend.backward
        def backward(*args,**kwargs):
            self.work['backwards_attempted']=self.work.get('backwards_attempted',0)+1
            result=ordinary_backward(*args,**kwargs)
            self.work['backwards_completed']=self.work.get('backwards_completed',0)+1
            return result
        self.backend.backward=backward

    def check_installed_arm(self):
        from experiments.mathqa_fixed_lr_transfer.training import FORBIDDEN_CONTROLLERS
        model=self.bundle.model
        if (model.objective!='osd_first_error' or not isinstance(model.first_error_redistribution,KPlusRedistribution)
            or model.prefix_allocation.contract!=prefix_allocation_contract(.25)
            or not getattr(model,'repair_conditional_kl',False)
            or (getattr(model,'continuation_repair',None) is not None)!=(self.arm=='TAIL')):
            raise ValueError('Full-run algorithm installation differs')
        if self.arm=='TAIL' and not isinstance(model.continuation_repair,ContinuationRepair):
            raise ValueError('Frozen TAIL formula changed')
        allowed={'prefix_allocation'}
        if any(getattr(model,key,None) is not None for key in set(FORBIDDEN_CONTROLLERS)-allowed):
            raise ValueError('Foreign experimental controller is active')

    def _append_observation(self,name,value):
        if name=='RESPONSE_OBSERVATIONS.jsonl':
            value['prefix_allocation']=deepcopy(self.bundle.model.prefix_allocation.last_summary)
            extension=getattr(self.bundle.model,'continuation_repair',None)
            if extension is not None:
                value['continuation_repair']={k:float(v.detach().item()) for k,v in extension.last_summary.items()}
            value['effective_components']=component_logging(value)
        return native.FullKTrainer._append_observation(self,name,value)

    def validation_arguments(self):
        return dict(arm=self.arm,binding=self.binding,config_text=self.config_text,
            original_config_text=self.original_config_text,expected_contract=self.contract,
            expected_recipe=self.backend.optimizer.recipe,expected_cuda_rng_states=self.expected_cuda_rng_states)

    def save_initial(self,comparator_path=None):
        destination=self.output_root/'INITIAL_WITNESS.pt'
        if destination.exists():raise FileExistsError(destination)
        with preserve_rng():
            state=native.live_state(self)
            checked=validate_snapshot(state,**self.validation_arguments(),completed_event=0)
            self.check_installed_arm();_export_equal(state,self.cfg.model.draft_checkpoint_path)
            witness=initial_witness(state);save_torch(destination,witness)
            if comparator_path is not None:
                checked['paired_initialization']=compare_initial_witnesses(
                    torch.load(comparator_path,map_location='cpu',weights_only=False),witness)
            if differences(witness,torch.load(destination,map_location='cpu',weights_only=False)):
                raise ValueError('Initial witness serialization differs')
            native.verify_live(self,state)
            checked.update(witness_path=str(destination),export_exact=True,algorithm_installed_before_witness=True)
            write_json(self.output_root/'INITIAL_CHECK.json',checked)
        return checked

    def _load_scientific(self,state):
        self.backend.optimizer.zero_grad();self.core._ratio_totals.clear();self.core._sum_totals.clear()
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'],strict=True)
        device=next(self.bundle.draft_model.parameters()).device
        self.backend.load_state_dict(isolated_backend_for_load(state['backend'],device))
        self.global_step,self.presentations=state['global_step'],state['presentations']
        self.events=deepcopy(state['events']);self.core._micro=self.presentations
        self._presentation_schedule=None;restore_rng(state['rng'])

    def _restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        validate_snapshot(state,**self.validation_arguments(),completed_event=self.completed_event)
        self.check_installed_arm();self._load_scientific(state)
        native.verify_live(self,state,self.output_root/'RESUME_CHECK.json')

    def _prepare_event(self,event_id,feature_paths,passes):
        number=len(self.events)+1
        if (self.qualification_only or not 1<=number<=58 or event_id!=f'event_{number:03d}' or
            len(feature_paths)!=128 or passes!=2 or self.global_step!=(number-1)*128 or self.presentations!=(number-1)*256):
            raise ValueError('Only the next complete own BASE/TAIL event is allowed')
        if not self.events and not (self.output_root/'INITIAL_WITNESS.pt').exists():
            raise ValueError('Authenticate released initialization before feedback/training')
        self._active_event=event_id;self.response_records.clear();self.window_records.clear();self.work.clear()
        features=SpecForgeEventTrainer._prepare_event(self,event_id,feature_paths,passes)
        previous={x['record_id'] for e in self.events for x in e['features']}
        if previous&{x['record_id'] for x in features}:raise ValueError('Own feedback repeats an earlier request')
        return features

    def _checkpoint_state(self,event):
        event.update(method=self.arm,dataset='OpenCodeInstruct',**{CONTRACT_KEY:experiment_contract(self.arm,self.binding)})
        validate_event_receipt(event,arm=self.arm,binding=self.binding,event_number=len(self.events)+1,require_completed=False)
        state=SpecForgeEventTrainer._checkpoint_state(self,event)
        write_json(self.events_root/event['event_id']/'RECOVERY_SIZE_PLAN.json',recovery_size_plan(state))
        return state

    def train_event(self,event_id,feature_paths,passes=2,*,reference_check=None,presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:raise ValueError('Full run uses natural own-feedback sampling')
        result=SpecForgeEventTrainer.train_event(self,event_id,feature_paths,passes)
        validate_event_receipt(result,arm=self.arm,binding=self.binding,event_number=len(self.events),trainer_status=self.status())
        validate_recovery_file(self.latest);self.completed_event=len(self.events)
        return result

    def verify_boundary(self,export_path=None,receipt_path=None):
        started=time.perf_counter()
        with preserve_rng():
            state=torch.load(self.latest,map_location='cpu',weights_only=False,mmap=True)
            result=validate_snapshot(state,**self.validation_arguments(),completed_event=len(self.events))
            self.check_installed_arm()
            if export_path is not None:_export_equal(state,export_path)
            result.update(live=native.verify_live(self,state),export_exact=export_path is not None,
                serialized_size=validate_recovery_file(self.latest),wall_s=time.perf_counter()-started)
        if receipt_path is not None:write_json(receipt_path,result)
        return result

    def import_qualification_parent(self,state,authentication):
        """Only disposable parity testing may import the authentic PR65 fixture."""
        from experiments.frontier_preservation_pilot.fork import scientific_state
        if not self.qualification_only or authentication.get('status')!='passed':
            raise ValueError('Historical state import is qualification-only')
        self._qualification_origin=dict(authentication)
        self._load_scientific(state)
        mismatch=differences(scientific_state(native.live_state(self)),scientific_state(state))
        if mismatch:
            raise ValueError('Disposable original scientific state did not restore exactly: '+repr(mismatch[:8]))
        self.check_installed_arm()

    def run_pair(self,pair,pass_index=0,ordinal=0):
        from experiments.bounded_prefix_allocation.training import CandidateTrainer
        if not self.qualification_only:raise ValueError('Disposable pair API cannot train production')
        return CandidateTrainer.run_pair(self,pair,pass_index,ordinal)

    def save_qualification(self,path):
        from experiments.prefix_ranking.qualify import cpu_copy
        if not self.qualification_only or self.core.accumulation_remainder:
            raise ValueError('Disposable serialization requires an empty accumulation boundary')
        self.check_installed_arm()
        state=cpu_copy(native.live_state(self))
        metadata=dict(schema=QUALIFICATION_SCHEMA,arm=self.arm,
            origin=deepcopy(self._qualification_origin),production_recovery=False,
            active_event=self._active_event,response_records=deepcopy(self.response_records),
            window_records=deepcopy(self.window_records),anchor_observations=deepcopy(self._anchor_observations),
            model_costs=deepcopy(self.bundle.model.costs),work=deepcopy(self.work),
            prefix_summary=deepcopy(self.bundle.model.prefix_allocation.last_summary),
            continuation_summary=deepcopy(getattr(self.bundle.model.continuation_repair,'last_summary',None)))
        save_torch(path,dict(state,qualification_state=metadata))
        return dict(schema=QUALIFICATION_SCHEMA,path=str(path),production_recovery=False,
            global_step=self.global_step,presentations=self.presentations)

    def restore_qualification(self,path,*,runtime=None):
        if not self.qualification_only:raise ValueError('Disposable state cannot resume production')
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        metadata=payload.pop('qualification_state')
        origin=metadata['origin']
        if (metadata['schema']!=QUALIFICATION_SCHEMA or metadata['arm']!=self.arm or
            metadata['production_recovery'] is not False or origin!=self._qualification_origin or
            payload['config_text']!=self.config_text or differences(payload['algorithm_contract'],self.contract) or
            payload['global_step'] not in (origin['global_step'],origin['global_step']+1) or
            payload['presentations']!=2*payload['global_step'] or len(payload['events'])!=1):
            raise ValueError('Foreign or fabricated disposable qualification state')
        self.check_installed_arm();self._load_scientific(payload)
        self.response_records=deepcopy(metadata['response_records']);self.window_records=deepcopy(metadata['window_records'])
        self._anchor_observations[:]=deepcopy(metadata['anchor_observations']);self._active_event=metadata['active_event']
        self.bundle.model.costs=deepcopy(metadata['model_costs']);self.work=deepcopy(metadata['work'])
        self.bundle.model.prefix_allocation.last_summary=deepcopy(metadata['prefix_summary'])
        if self.arm=='TAIL':self.bundle.model.continuation_repair.last_summary=deepcopy(metadata['continuation_summary'])
        if runtime is not None:
            from experiments.functional_step_geometry.qualify import restore_runtime
            restore_runtime(self.bundle.model,runtime)
        result=native.verify_live(self,payload)
        return dict(result,schema=QUALIFICATION_SCHEMA,production_recovery=False)
