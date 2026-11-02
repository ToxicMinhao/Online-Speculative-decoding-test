"""Persistent native K training with explicit two-arm, 58-event boundaries.

Only experiment contracts, state checks and observations are added here. The
model, first-error KL, sampler, accumulation, Adam and exporter stay native.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import time

from experiments.ok_continuation.resume import differences, _finite
from experiments.ok_coding.training import _export_equal
from experiments.o_half_matched_lr.training import native_algorithm_contract
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer, SCHEMA

LEARNING_RATES = {'K': 3e-4, 'K_HALF': 1.5e-4}
CONTRACT_KEY = 'full_k_gsm8k_v1'
OBJECTIVE = 'osd_first_error'


def experiment_contract(arm, binding):
    if arm not in LEARNING_RATES or not isinstance(binding, str) or len(binding) != 64:
        raise ValueError('Require K/K_HALF and the sealed SHA-256 campaign binding')
    if any(character not in '0123456789abcdef' for character in binding):
        raise ValueError('Malformed campaign binding')
    return dict(schema=CONTRACT_KEY, arm=arm, training_seed=17, binding=binding,
                learning_rate=LEARNING_RATES[arm], maximum_events=58,
                initialization='independent_released_seed17_empty_adam',
                feedback='own_fresh_native_responses', reference=None)


def validate_config_text(config_text, arm, original_config_text):
    """Allow LR and output identity changes from the sealed full O recipe."""
    import yaml
    from experiments.ok_coding.training import validate_config_text as original_gate
    source, current = yaml.safe_load(original_config_text), yaml.safe_load(config_text)
    original_gate(original_config_text, 17)
    if (arm not in LEARNING_RATES or current['training']['learning_rate'] != LEARNING_RATES[arm]
            or source['online_feedback']['max_events'] != 58):
        raise ValueError('Full K arm LR or inherited 58-event horizon differs')
    checked = deepcopy(current)
    checked['training']['learning_rate'] = 3e-4
    for key in ('run_id', 'output_dir'):
        if key in source:
            checked[key] = source[key]
        else:
            checked.pop(key, None)
    if differences(checked, source):
        raise ValueError('Full K configuration differs beyond LR/run/output identity')


def validate_event_receipt(receipt, *, arm, event_number, require_completed=True, trainer_status=None):
    """Validate actual event numbers through 58 without weakening old gates."""
    if arm not in LEARNING_RATES or type(event_number) is not int or not 1 <= event_number <= 58:
        raise ValueError('Require an authorized K/K_HALF event 1..58')
    expected = dict(event_id=f'event_{event_number:03d}', objective=OBJECTIVE,
        passes=2, response_presentations=256, optimizer_steps=128,
        start_step=(event_number-1)*128, end_step=event_number*128, reduction='response_mean')
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Event identity, objective or budget differs')
    if require_completed and receipt.get('status') != 'completed':
        raise ValueError('Event is incomplete')
    features, records = receipt['features'], receipt['records']
    ids = [item['record_id'] for item in features]
    if len(ids) != 128 or len(set(ids)) != 128 or len(records) != 256:
        raise ValueError('Require 128 distinct responses and 256 presentations')
    for index, row in enumerate(records):
        stepped = index % 2 == 1
        if (row['pass'] != index // 128 or type(row['optimizer_stepped']) is not bool
                or row['optimizer_stepped'] != stepped
                or row['optimizer_step_after'] != expected['start_step'] + (index+1)//2
                or row['learning_rate'] != (LEARNING_RATES[arm] if stepped else None)):
            raise ValueError('Natural pass, native Adam boundary or learning rate differs')
        if len(row['sampled_anchors']) != 1:
            raise ValueError('Require exactly one natural anchor draw per presentation')
        draw = row['sampled_anchors'][0]
        positions, keep = draw['positions'], draw['keep_mask']
        if (len(positions) != 1 or len(keep) != 1 or not 0 < len(positions[0]) <= 512
                or len(positions[0]) != len(keep[0])
                or any(type(item) is not int or item < 0 for item in positions[0])
                or any(type(item) is not bool for item in keep[0])):
            raise ValueError('Malformed qualified anchor draw')
        if any(not math.isfinite(float(value)) for value in row['metrics'].values()):
            raise ValueError('Nonfinite native training metric')
    for pass_index in (0, 1):
        if sorted(row['record_id'] for row in records[128*pass_index:128*(pass_index+1)]) != sorted(ids):
            raise ValueError('Each pass must present every own response once')
    if trainer_status is not None:
        status = dict(global_step=128*event_number, presentations=256*event_number,
            completed_events=[f'event_{n:03d}' for n in range(1,event_number+1)],
            accumulation_remainder=0, objective=OBJECTIVE, horizon=7424)
        if any(trainer_status.get(key) != value for key, value in status.items()):
            raise ValueError('Trainer does not match its complete own event')
    return dict(status='passed', arm=arm, **expected, anchor_draws_checked=256,
                historical_anchors_forced=False, extra_optimizer_steps=0)


def validate_snapshot(state, *, arm, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe,
                      expected_cuda_rng_states=1):
    """Read-only exact own-state validation; no construction or RNG restoration."""
    import random
    import numpy as np
    import torch
    from experiments.osd_components.checkpoints import EXPECTED_KEYS
    if type(completed_event) is not int or not 0 <= completed_event <= 58:
        raise ValueError('Complete full K boundary must be event 0..58')
    validate_config_text(config_text, arm, original_config_text)
    required = {'event_schema','draft_state_dict','backend','rng','global_step','presentations',
        'events','strategy','run_id','world_size','config_text','algorithm_contract',
        'reduction','accumulation_steps','horizon'}
    if set(state) != required or set(state['backend']) != {'optimizer'}:
        raise ValueError('Unexpected or missing scientific state fields')
    contract = native_algorithm_contract(expected_contract)
    if (contract.get(CONTRACT_KEY) != experiment_contract(arm,binding)
            or contract['online_target_objective']['name'] != OBJECTIVE
            or differences(state['algorithm_contract'],contract)):
        raise ValueError('Foreign full K arm, objective or scientific contract')
    expected = dict(event_schema=SCHEMA, strategy='dflash', world_size=1,
        config_text=config_text, reduction='response_mean', accumulation_steps=2, horizon=7424,
        global_step=128*completed_event, presentations=256*completed_event)
    if any(state.get(key) != value for key,value in expected.items()) or len(state['events']) != completed_event:
        raise ValueError('Own complete state identity or counters differ')
    seen = set()
    for number,event in enumerate(state['events'],1):
        validate_event_receipt(event,arm=arm,event_number=number,require_completed=False)
        ids = {item['record_id'] for item in event['features']}
        if seen.intersection(ids):
            raise ValueError('Own feedback event reuses an earlier request identity')
        seen.update(ids)
    weights = state['draft_state_dict']
    if set(weights) != EXPECTED_KEYS:
        raise ValueError('Require the exact 58 draft parameter names')
    for name,value in weights.items():
        if not isinstance(value,torch.Tensor) or value.dtype != torch.bfloat16 or not value.numel():
            raise ValueError('Invalid native BF16 draft tensor: '+name)
        _finite(value,name)
    optimizer = state['backend']['optimizer']
    recipe,saved = optimizer['recipe'],optimizer['optimizer_state_dict']
    if differences(recipe,expected_recipe) or recipe['type'] != 'direct_bf16_adamw' or recipe['version'] != 1:
        raise ValueError('Native optimizer recipe or parameter ordering differs')
    if len(saved['param_groups']) != 1:
        raise ValueError('Require one native Adam group')
    group = saved['param_groups'][0]
    options = {key:value for key,value in group.items() if key != 'params'}
    if (differences(options,recipe['optimizer_options']) or recipe['max_grad_norm'] != 1.
            or options['lr'] != LEARNING_RATES[arm] or options['weight_decay'] != .01
            or options['betas'] != (.9,.999) or options['eps'] != 1e-8 or options['amsgrad'] is not False):
        raise ValueError('Qualified native Adam hyperparameters differ')
    parameters = recipe['parameters']
    ids = group['params']
    if (len(parameters) != len(ids) or len(ids) != 58 or len(set(ids)) != 58
            or {item[0] for item in parameters} != set(weights)
            or set(saved['state']) != (set(ids) if completed_event else set())):
        raise ValueError('Native Adam parameter or moment membership differs')
    for (name,shape,dtype),parameter_id in zip(parameters,ids):
        weight = weights[name]
        if tuple(shape) != tuple(weight.shape) or dtype != str(weight.dtype):
            raise ValueError('Optimizer binding differs: '+name)
        if not completed_event:
            continue
        moments = saved['state'][parameter_id]
        if set(moments) != {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Incomplete native Adam state: '+name)
        counter = moments['step']
        if not isinstance(counter,torch.Tensor) or counter.numel()!=1 or counter.item()!=state['global_step']:
            raise ValueError('Native Adam counter differs: '+name)
        for key in ('exp_avg','exp_avg_sq'):
            value = moments[key]
            if value.dtype != torch.bfloat16 or value.shape != weight.shape:
                raise ValueError('Native Adam moment precision/shape differs: '+name)
            _finite(value,name+'.'+key)
    rng = state['rng']
    if (set(rng) != {'python','numpy','torch','cuda'} or expected_cuda_rng_states not in (0,1)
            or not isinstance(rng['cuda'],list) or len(rng['cuda']) != expected_cuda_rng_states):
        raise ValueError('Require complete qualified RNG state')
    for value in [rng['torch'],*rng['cuda']]:
        if not isinstance(value,torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel():
            raise ValueError('Malformed Torch RNG payload')
    random.Random(0).setstate(rng['python'])
    np.random.RandomState(0).set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch'])
    return dict(status='passed',arm=arm,completed_event=completed_event,
        global_step=state['global_step'],presentations=state['presentations'],
        adam_state_entries=len(saved['state']),cuda_rng_states=len(rng['cuda']),
        extra_model_passes=0,extra_optimizer_steps=0)


def live_state(trainer):
    full = trainer.backend.state_dict()
    return dict(event_schema=SCHEMA,
        draft_state_dict=trainer.strategy.checkpoint_state_filter(full['model']),
        backend={'optimizer':full['optimizer']},rng=capture_rng(),
        global_step=trainer.global_step,presentations=trainer.presentations,events=trainer.events,
        strategy='dflash',run_id=trainer.cfg.run_id,world_size=1,config_text=trainer.config_text,
        algorithm_contract=trainer.contract,reduction=trainer.reduction,
        accumulation_steps=trainer.accumulation,horizon=trainer.horizon)


def verify_live(trainer,state,receipt_path=None):
    actual = live_state(trainer)
    mismatch = differences(actual,state)
    model = trainer.bundle.model
    if (trainer.closed or trainer.failed or trainer.core.accumulation_remainder
            or trainer.core._micro != trainer.presentations or trainer.core._ratio_totals
            or trainer.core._sum_totals or trainer._presentation_schedule is not None
            or not model.training or model.reference is not None or model.ranking_enabled
            or model.objective != OBJECTIVE
            or any(parameter.grad is not None for parameter in trainer.bundle.draft_model.parameters())):
        mismatch.append('unclean_live_boundary')
    receipt = dict(status='passed' if not mismatch else 'failed',exact_resume=not mismatch,
        mismatches=mismatch[:32],global_step=trainer.global_step,presentations=trainer.presentations,
        no_rng_reset=True,extra_model_passes=0,extra_optimizer_steps=0)
    if receipt_path is not None:
        write_json(receipt_path,receipt)
    if mismatch:
        raise ValueError('Live full K state differs: '+str(mismatch[:8]))
    return receipt


def compare_initial_states(first,second):
    """Both native initializations already validated; compare the allowed difference."""
    values=[]
    for state in (first,second):
        optimizer=deepcopy(state['backend']['optimizer'])
        optimizer['recipe']['optimizer_options']['lr']=3e-4
        optimizer['optimizer_state_dict']['param_groups'][0]['lr']=3e-4
        contract = deepcopy(state['algorithm_contract'])
        contract.pop(CONTRACT_KEY)
        values.append(dict(weights=state['draft_state_dict'],optimizer=optimizer,rng=state['rng'],
            contract=contract,global_step=state['global_step'],presentations=state['presentations'],events=state['events']))
    mismatch=differences(*values)
    if mismatch:
        raise ValueError('Independent arm initialization differs: '+str(mismatch[:8]))
    return dict(status='passed',exact_released_weights_empty_adam_rng=True,
                allowed_difference='LR and arm/run/output metadata only',rng_overwritten=False)


class FullKTrainer(SpecForgeEventTrainer):
    """One persistent native trainer per own-feedback stream."""

    def __init__(self,config_path,output_root,*,arm,binding,original_config_text,
                 resume_state=None,completed_event=0,expected_cuda_rng_states=1):
        if type(completed_event) is not int or not 0 <= completed_event <= 58:
            raise ValueError('Require complete full K boundary0..58')
        if (completed_event>0) != (resume_state is not None):
            raise ValueError('Continuation requires its exact own recovery state')
        self.arm,self.binding=arm,binding
        self.original_config_text=original_config_text
        self.expected_cuda_rng_states=expected_cuda_rng_states
        self.completed_event=completed_event
        self.response_records,self.window_records=[],[]
        self.work = {}
        self._work_handles = []
        experiment_contract(arm,binding)
        super().__init__(config_path,output_root,resume_state=resume_state,
                         objective=OBJECTIVE,reduction='response_mean',ranking_lambda=0.)

    def _validate_config(self):
        super()._validate_config()
        validate_config_text(self.config_text,self.arm,self.original_config_text)

    def _build_runtime(self):
        super()._build_runtime()
        self.contract[CONTRACT_KEY]=experiment_contract(self.arm,self.binding)
        self._install_observations()

    def validation_arguments(self):
        return dict(arm=self.arm,binding=self.binding,config_text=self.config_text,
            original_config_text=self.original_config_text,expected_contract=self.contract,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _restore(self,path):
        import torch
        state=torch.load(path,map_location='cpu',weights_only=False)
        validate_snapshot(state,**self.validation_arguments(),completed_event=self.completed_event)
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'],strict=True)
        self.backend.load_state_dict(state['backend'])
        self.global_step,self.presentations=state['global_step'],state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        restore_rng(state['rng'])
        verify_live(self,state,receipt_path=self.output_root/'RESUME_CHECK.json')

    def save_initial(self,comparator_path=None):
        import torch
        destination=self.output_root/'initial_state.pt'
        if destination.exists() or self.global_step or self.presentations or self.events:
            raise ValueError('Initial snapshot must be genuinely new and empty')
        if (self.arm == 'K_HALF') != (comparator_path is not None):
            raise ValueError('K_HALF must compare against the new K initial state')
        with preserve_rng():
            state=live_state(self)
            receipt=validate_snapshot(state,**self.validation_arguments(),completed_event=0)
            _export_equal(state,self.cfg.model.draft_checkpoint_path)
            if comparator_path is not None:
                first=torch.load(comparator_path,map_location='cpu',weights_only=False)
                if first['algorithm_contract'][CONTRACT_KEY] != experiment_contract('K',self.binding):
                    raise ValueError('Initial comparator is not this campaign\'s released K')
                receipt['paired_initialization']=compare_initial_states(first,state)
            save_torch(destination,state)
            decoded=torch.load(destination,map_location='cpu',weights_only=False)
            verify_live(self,decoded)
            write_json(self.output_root/'INITIAL_CHECK.json',dict(receipt,export_exact=True,state_path=str(destination)))
        return dict(receipt,state_path=str(destination))

    def verify_boundary(self,export_path=None,receipt_path=None):
        import torch
        started=time.perf_counter()
        with preserve_rng():
            state=torch.load(self.latest,map_location='cpu',weights_only=False)
            receipt=validate_snapshot(state,**self.validation_arguments(),completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state,export_path)
                receipt['export_exact']=True
            receipt['live']=verify_live(self,state)
        receipt['wall_s']=time.perf_counter()-started
        if receipt_path is not None:
            write_json(receipt_path,receipt)
        return receipt

    def _append_observation(self,name,value):
        with (self.events_root/self._active_event/name).open('a') as stream:
            stream.write(json.dumps(value,allow_nan=True)+'\n')

    def _install_observations(self):
        def count(name):
            def observed(_module,_inputs):
                self.work[name]=self.work.get(name,0)+1
            return observed
        model=self.bundle.model
        self._work_handles.append(model.draft_model.register_forward_pre_hook(count('draft_forwards')))
        self._work_handles.append(model.lm_head.register_forward_pre_hook(count('vocabulary_head_calls')))
        for layer in model.draft_model.layers:
            self._work_handles.append(layer.register_forward_pre_hook(count('layer_forwards')))
        self._native_forward=self.author_strategy.forward_loss
        self._native_step=self.backend.optimizer.step
        self._native_chunk=model._chunk_loss
        self._native_end=model.end_event
        def scalar(value):
            return float(value.detach().item()) if hasattr(value,'detach') else float(value)
        def observed_forward(batch,ctx=None):
            output=self._native_forward(batch,ctx)
            index=len(self.response_records)
            record=dict(scope='one_original_response_forward',record_id=str(batch.sample_ids[0]),
                event_id=self._active_event,pass_index=index//128,presentation_ordinal=index,
                sampled_anchors=deepcopy(self._anchor_observations),loss=scalar(output.loss),
                ratio_metrics={key:[scalar(a),scalar(b)] for key,(a,b) in output.ratio_metrics.items()},
                sum_metrics={key:scalar(value) for key,value in output.sum_metrics.items()})
            self.response_records.append(record)
            self._append_observation('RESPONSE_OBSERVATIONS.jsonl',record)
            self.work['response_forwards']=self.work.get('response_forwards',0)+1
            return output
        def observed_step():
            result=self._native_step()
            recent=self.response_records[-2:]
            record=dict(scope='optimizer_window',event_id=self._active_event,
                window=len(self.window_records)+1,global_step=self.global_step+1,
                record_ids=[row['record_id'] for row in recent],gradient_norm=scalar(result),
                learning_rate=self.backend.optimizer.get_learning_rate(),
                mean_response_loss=sum(row['loss'] for row in recent)/2)
            self.window_records.append(record)
            self._append_observation('WINDOW_OBSERVATIONS.jsonl',record)
            return result
        def observed_chunk(*args,**kwargs):
            self.work['objective_chunks']=self.work.get('objective_chunks',0)+1
            return self._native_chunk(*args,**kwargs)
        def observed_end():
            return dict(self._native_end(),work=dict(self.work),reference_operations=0)
        self.author_strategy.forward_loss=observed_forward
        self.backend.optimizer.step=observed_step
        model._chunk_loss,model.end_event=observed_chunk,observed_end

    def _prepare_event(self,event_id,feature_paths,passes):
        number=len(self.events)+1
        if (not 1<=number<=58 or event_id!=f'event_{number:03d}'
                or self.global_step!=(number-1)*128 or self.presentations!=(number-1)*256
                or len(feature_paths)!=128 or type(passes) is not int or passes!=2):
            raise ValueError('Require the next complete own 128-response/two-pass event')
        if not self.events and not (self.output_root/'initial_state.pt').exists():
            raise ValueError('Save the genuine initial state before serving or training')
        self._active_event=event_id
        self.response_records.clear()
        self.window_records.clear()
        self.work.clear()
        features=super()._prepare_event(event_id,feature_paths,passes)
        previous={item['record_id'] for event in self.events for item in event['features']}
        if previous.intersection(item['record_id'] for item in features):
            raise ValueError('A later event cannot reuse an earlier own-feedback request')
        return features

    def _train_passes(self,features,passes):
        rows=super()._train_passes(features,passes)
        if len(rows)!=len(self.response_records) or len(self.window_records)!=128:
            raise ValueError('Original forward/window observation accounting differs')
        for row,observed in zip(rows,self.response_records):
            if (row['record_id']!=observed['record_id'] or row['pass']!=observed['pass_index']
                    or row['sampled_anchors']!=observed['sampled_anchors']):
                raise ValueError('Response observation identity or anchor binding differs')
            row['response_observation']=observed
        return rows

    def train_event(self,event_id,feature_paths,passes=2,*,reference_check=None,presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('Own-feedback full K has no reference probe or forced historical schedule')
        result=super().train_event(event_id,feature_paths,passes)
        validate_event_receipt(result,arm=self.arm,event_number=len(self.events),trainer_status=self.status())
        self.completed_event=len(self.events)
        return result

    def close(self):
        if self.closed:
            return
        for handle in self._work_handles:
            handle.remove()
        self._work_handles.clear()
        if hasattr(self,'_native_forward'):
            self.author_strategy.forward_loss=self._native_forward
            self.backend.optimizer.step=self._native_step
            self.bundle.model._chunk_loss=self._native_chunk
            self.bundle.model.end_event=self._native_end
        return super().close()


@contextmanager
def preserve_training_rng(trainer):
    """Intermediate evaluation is external; its parent process cannot alter draws."""
    before=capture_rng()
    mode=trainer.bundle.model.training
    counters=(trainer.global_step,trainer.presentations,len(trainer.events))
    try:
        yield
    finally:
        restore_rng(before)
        trainer.bundle.model.train(mode)
        if counters!=(trainer.global_step,trainer.presentations,len(trainer.events)):
            raise ValueError('Intermediate evaluation changed training counters')
