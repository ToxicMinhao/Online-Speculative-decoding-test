"""Explicit 0→8 L8 control on the unchanged qualified model/runtime.

Imports are CPU-only. The historical trainers are not patched. One fresh lower-LR
trainer exports one genuinely registered L8 checkpoint, evaluates fixed losses,
and exits (including its children) before any natural L8 generation.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap
import time

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
_spec = importlib.util.spec_from_file_location('_lower_lr_qualified_diagnostic',
    HERE.parent / 'dflash_kl_native_diagnostic/native.py')
kl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kl
_spec.loader.exec_module(kl)
curve, base, followup = kl.curve, kl.base, kl.followup
require, sha, digest, write_json = base.require, base.sha, base.object_hash, base.write_json
PARAMETERS, PARAMETER_BYTES = 1048626432, 2
CHECKPOINT_METADATA_BYTES = 64 * 2**20
GROUP_EVIDENCE_BYTES = 256 * 2**20
MIN_RESERVE_BYTES = 8 * 2**30
STOP_IDS = kl.STOP_IDS


def source_pins():
    pins = kl.source_pins()
    for pattern in ('*.py', '*.json', 'requirements*.txt'):
        for path in sorted(HERE.glob(pattern)):
            if not path.name.startswith('test_'):
                pins[str(path.relative_to(HERE.parent))] = sha(path)
    return dict(sorted(pins.items()))


def check_sources(config):
    require(config.get('lower_lr_source_sha256') == source_pins(), 'Reviewed lower-LR source changed')
    require('diagnostic_source_sha256' not in config or
            config['diagnostic_source_sha256'] == kl.source_pins(), 'Qualified PR10 source changed')
    config['diagnostic_source_sha256'] = kl.source_pins()
    kl.check_sources(config)


def campaign_storage_bound():
    # Six retained BF16 weights, one conservative full export/staging overlap,
    # metadata allowance, and compact evidence. No disk optimizer-state export.
    model_bytes = PARAMETERS * PARAMETER_BYTES
    checkpoint_bytes = model_bytes + CHECKPOINT_METADATA_BYTES
    return dict(parameter_count=PARAMETERS, parameter_bytes=PARAMETER_BYTES,
        model_tensor_bytes=model_bytes, final_checkpoints=6,
        final_checkpoint_bytes=6 * checkpoint_bytes,
        export_staging_overlap_bytes=checkpoint_bytes,
        compact_evidence_bytes=6 * GROUP_EVIDENCE_BYTES,
        optimizer_state_exports=0, intermediate_checkpoints=0,
        planned_write_bytes=7 * checkpoint_bytes + 6 * GROUP_EVIDENCE_BYTES)


def storage_preflight(config):
    root = Path(config['output_root']).resolve()
    while not root.exists():
        root = root.parent
    reserve = config.get('reserve_bytes', MIN_RESERVE_BYTES)
    require(type(reserve) is int and reserve >= MIN_RESERVE_BYTES, 'Preserve the 8 GiB storage reserve')
    receipt = campaign_storage_bound()
    required = receipt['planned_write_bytes'] + reserve
    free = shutil.disk_usage(root).free
    require(free >= required, 'Six final L8 weights, staging, compact evidence and reserve exceed free space')
    return dict(status='passed', **receipt, free_bytes=free, required_bytes=required, reserve_bytes=reserve,
        formula='7*(1048626432*2 + 64MiB) + 6*256MiB + reserve; no old artifact deletion')


def worker_command(config):
    command = followup._original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Qualified worker transport changed')
    command[-1] = str(HERE / 'native.py')
    return command


base.worker_command = worker_command
close_owned_worker = kl.close_owned_worker


def admit_checkpoint(config, checkpoint):
    require(checkpoint.get('checkpoint_id') == 'L8', 'Only a genuine L8 checkpoint is admitted')
    expected = Path(config['episode_out']) / 'checkpoints/L8'
    require(Path(checkpoint['path']) == expected and checkpoint.get('group_id') == config['episode_id'],
            'L8 belongs to a different group or checkpoint path')
    require(checkpoint.get('optimizer_steps') == 8 and checkpoint.get('learning_rate') == 3e-5,
            'L8 checkpoint learning trajectory differs')
    base.verify_checkpoint(checkpoint)
    return 'L8'


class NativeBackend(base.NativeBackend):
    def __init__(self, config):
        super().__init__(config)
        self.training_started = self.training_closed = False
        self.retired_trainer_pid = None
        self.served = set()

    def start_group(self, original_group):
        require(self.episode_id is None, 'Close prior L8 group first')
        check_sources(self.config)
        base.check_environment(self.config)
        dependencies = base.check_dependencies(self.config['workspace_root'])
        require(self.config['runtime_source_sha256'] == dependencies['runtime_source_sha256'], 'Qualified runtime changed')
        require(self.config.get('output_cap', 1024) == 1024 and self.config.get('block_size', 10) == 10,
                'Keep greedy block10 and 1024-or-EOS')
        group = copy.deepcopy(original_group)
        group_id = group['group_id']
        require(Path(group_id).name == group_id and group_id not in ('.', '..'), 'Invalid group identifier')
        require(group['status'] == 'complete' and len(group['feedback']) == len(group['support']) == 4,
                'Four original completed frozen feedback records required')
        require(set(group['reference_pairs']) == {'source', 'heldout'} and
                all(len(group['reference_pairs'][p]) == 4 for p in ('source','heldout')), 'Bind all eight PR10 pairs')
        root = Path(self.config['output_root']).resolve(); root.mkdir(parents=True, exist_ok=True)
        budget = 2 * (PARAMETERS * PARAMETER_BYTES + CHECKPOINT_METADATA_BYTES) + GROUP_EVIDENCE_BYTES
        reserve = self.config.get('reserve_bytes', MIN_RESERVE_BYTES)
        require(reserve >= MIN_RESERVE_BYTES and shutil.disk_usage(root).free >= reserve + budget,
                'L8 group writes and export overlap would cross storage reserve')
        out = root / group_id
        require(not out.exists(), 'Never overwrite an L8 attempt')
        models_path = base.checked_binding(self.config['model_bindings'])
        require(sha(models_path) == dependencies['qualified_model_bindings']['sha256'], 'Qualified model registry changed')
        models = json.loads(models_path.read_text())
        target = next(v for v in models['targets'] if v['target_id'] == 'T0')
        released = models['released_draft']
        frozen = dict(path=released['path'], files=released['checkpoint_files'], checkpoint_id='F',
                      checkpoint_identity_sha256=digest(released['checkpoint_files']))
        require(group['checkpoints']['F'] == frozen, 'Original released checkpoint identity changed')
        base.verify_checkpoint(frozen)
        for name, bound in target['checkpoint_files'].items():
            require(Path(name).name == name, 'Target file escapes directory')
            base.checked_binding({'path': str(Path(target['model_path']) / name), **bound})
        for item in group['support']:
            path = base.checked_binding(item['metadata_binding'])
            saved = json.loads(path.read_text())
            require(all(item.get(k) == v for k,v in saved.items()) and item['episode_id'] == group_id and
                    item['capture_checkpoint'] == 'F' and
                    item['capture_checkpoint_sha256'] == frozen['checkpoint_identity_sha256'],
                    'Original support metadata or frozen origin changed')
        out.mkdir()
        cfg = {**self.config, 'episode_id':group_id, 'episode_out':str(out), 'target':target, 'released':frozen,
            'original_group':group, 'registry_path':str(models_path), 'registry_sha256':sha(models_path),
            'tokenizer_path':models.get('tokenizer_path',target['model_path']),
            'max_capture_bytes':self.config.get('max_capture_bytes',base.MAX_CAPTURE_BYTES),
            'effective_limits':dict(input_tokens=base.INPUT_LIMIT,total_context_tokens=32768,output_tokens=1024,
                proposal_block_tokens=10,chunked_prefill_tokens=8192,capture_memory_bound_bytes=base.capture_memory_bound()),
            'public_source_sha256':{name:sha(base.HERE/name) for name in ('native.py','training.py','native_dependencies.json')}}
        self.config,self.group,self.models,self.target,self.out,self.episode_id=cfg,group,models,target,out,group_id
        self.checkpoints={'F':frozen}; self.served=set()
        self.training_started=self.training_closed=False; self.retired_trainer_pid=None
        write_json(out/'NATIVE_CONFIG.json',cfg)
        return dict(status='passed',group_id=group_id,released=frozen,fresh_optimizer=True,seed=17,
            learning_rate=3e-5,training_steps=8,warmup_generations=0,model_worker_started=False)

    def train_group(self):
        require(self.episode_id is not None and not self.training_started and self.trainer is None,
                'Exactly one fresh 0→8 trajectory per group')
        self.training_started=True
        self.trainer=base._WorkerClient(self.config,'train',self.out/'training_worker')
        self.totals['startup_s']+=self.trainer.startup_s
        try:
            result=self.trainer.call('train_group',{})
        except BaseException as error:
            path=self.out/'training_worker/PARTIAL.json'
            if path.is_file():
                error.diagnostic_partial=json.loads(path.read_text())
                error.partial_binding=base.binding(path)
            raise
        result['trainer_startup_s']=self.trainer.startup_s
        checkpoint=result['checkpoint']; admit_checkpoint(self.config,checkpoint)
        self.checkpoints['L8']=checkpoint
        write_json(self.out/'TRAINING_RESULT.json',result)
        return result

    def assert_training_quiescent(self):
        require(self.trainer is None and self.training_closed, 'Stop training before native evaluation')
        require(self.retired_trainer_pid is None or not curve.process_group_alive(self.retired_trainer_pid),
                'Training child still alive; native evaluation blocked')
        return dict(status='passed',training_quiescent=True,terminated_process_group=self.retired_trainer_pid)

    def stop_training(self):
        require(self.training_started and self.trainer is not None and 'L8' in self.checkpoints,
                'Complete the L8 trajectory before normal training shutdown')
        worker=self.trainer
        receipt=close_owned_worker(worker)
        self.trainer=None; self.training_closed=True; self.retired_trainer_pid=worker.process.pid
        self.totals['shutdown_s']+=receipt['shutdown_s']
        receipt.update(self.assert_training_quiescent())
        write_json(self.out/'TRAINING_QUIESCENT.json',receipt)
        return receipt

    def serve_request(self, canonical, population):
        self.assert_training_quiescent()
        require(population in ('source','heldout'), 'Unknown request population')
        originals=[p['F']['native'] for p in self.group['reference_pairs'][population]]
        matches=[i for i,r in enumerate(originals) if r['request_id']==canonical['request_id']]
        require(len(matches)==1 and originals[matches[0]]==canonical, 'Saved request differs from bound PR10 input')
        key=(population,canonical['request_id'])
        require(key not in self.served, 'One natural L8 generation per request; no repetitions')
        self.served.add(key)
        result=super().serve(list(canonical['input_ids']),'L8',False,
            turn_index=matches[0]+1+(4 if population=='heldout' else 0))
        trace=json.loads(base.checked_binding(result['trajectory']).read_text())
        rounds,initial=curve.normalize_rounds(trace,result,canonical['input_ids'])
        calls=result['verification_calls']
        result.update(input_ids=list(canonical['input_ids']),checkpoint_id='L8',request_id=canonical['request_id'],
            group_id=self.episode_id,domain=self.group.get('domain'),role=canonical['role'],population=population,
            capture=False,feedback_capture_enabled=False,output_tokens=len(result['output_ids']),
            context_tokens=len(canonical['input_ids']),rounds=rounds,initial_output_tokens=initial,
            cap_hit=len(result['output_ids'])==1024,eos_hit=result['output_ids'][-1] in STOP_IDS,
            committed_progress_per_call=result['verification_committed_tokens']/calls if calls else None,
            emitted_tokens_per_verification_call=len(result['output_ids'])/calls if calls else None,
            timing_scope='Fresh L8 complete request only; no comparison against historical F/S8 timing.')
        return result

    def train_event(self,*args,**kwargs):
        raise ValueError('Use only the explicit L8 0→8 control path')

    def finish_group(self):
        errors=[]
        for name in ('worker','trainer'):
            worker=getattr(self,name)
            if worker is not None:
                try:
                    receipt=close_owned_worker(worker)
                    self.totals['shutdown_s']+=receipt['shutdown_s']
                except Exception as error:
                    errors.append(repr(error))
                setattr(self,name,None)
        result=dict(status='failed' if errors else 'passed',group_id=self.episode_id,**self.totals,errors=errors)
        if self.episode_id is not None:
            result['free_after_bytes']=shutil.disk_usage(self.out).free
            write_json(self.out/'NATIVE_FINISH.json',result)
        self.episode_id=None
        require(not errors,'Owned L8 workers failed shutdown')
        return result


def _l8_generate_method():
    """Extend only trace-version metadata; keep the actual native generation body."""
    original=base._ServingService.generate
    source=textwrap.dedent(inspect.getsource(original))
    before="{'F': 0, 'U1': 1, 'U2': 2}[self.active['checkpoint_id']]"
    after="{'F': 0, 'L8': 8}[self.active['checkpoint_id']]"
    require(source.count(before)==1,'Qualified trace-version registration changed')
    scope={}
    exec(compile(source.replace(before,after),original.__code__.co_filename,'exec'),base.__dict__,scope)
    return scope['generate']


class ServingService(base._ServingService):
    native_generate=_l8_generate_method()

    def __init__(self,config):
        check_sources(config)
        super().__init__(config)
        write_json(self.out/'LOWER_LR_SERVING_CONTRACT.json',dict(checkpoint_id='L8',native_trace_version=8,
            output_cap=1024,block_size=10,context_limit=32768,prefill_chunk_size=8192,
            feedback_capture=False,warmups=0,cache_policy='Cold every request and checkpoint switch'))

    def publish(self,checkpoint):
        admit_checkpoint(self.config,checkpoint)
        self.bridge._two_update_admitted.add(Path(checkpoint['path']))
        started=time.perf_counter()
        receipt=self.bridge.publish({'model_path':checkpoint['path'],'reset':True})
        require(receipt['status']=='passed' and receipt['success'] and
                receipt['helper_matches_current_layers'] and receipt['helper_norm_matches_current_layers'] and
                receipt['cache_after']['tree_total_size']==0 and
                self.bridge.state()['draft_model_path']==checkpoint['path'], 'Native L8 publication/cache invalidation failed')
        self.active=checkpoint; write_json(self.out/'ACTIVE_CHECKPOINT.json',checkpoint)
        return dict(status='passed',checkpoint_id='L8',activation_s=time.perf_counter()-started,native_receipt=receipt)

    def generate(self,body):
        require(body['capture'] is False and body['checkpoint_id']=='L8', 'Only capture-disabled natural L8 generation')
        result=self.native_generate(body)
        require(result['feedback'] is None and result['capture_costs'] is None,'Feedback capture was active')
        return result


class TrainerService:
    """Qualified loader with one explicit lower-LR trainer, no historical guards bypassed."""
    _frozen_versions=followup.TrainerService._frozen_versions
    check_frozen=followup.TrainerService.check_frozen
    teacher_features=followup.TrainerService.teacher_features
    close=followup.TrainerService.close

    def __init__(self,config):
        check_sources(config)
        self.config,self.out=config,Path(config['worker_out'])
        directory=Path(config['workspace_root'])/'research/target_variant_specialization_001/training_tools'
        sys.path.insert(0,str(directory))
        import native as qualified_native
        require(Path(qualified_native.__file__).resolve()==directory/'native.py','Qualified training loader collision')
        self.native=qualified_native
        self.runtime=qualified_native.load_runtime(17,1024)
        require(Path(qualified_native.CORE.AUTHOR).resolve()==Path(config['author_root']).resolve(),'Author source tree changed')
        args=copy.copy(self.runtime.default_args)
        require(config['released']['path']==str(qualified_native.CORE.DRAFT) and
                config['target']['model_path']==str(qualified_native.CORE.TARGET),'Qualified model pair changed')
        args.target_model_path=config['target']['model_path']
        args.initial_draft_path=args.draft_config_path=config['released']['path']
        args.max_length=args.sglang_context_length=32768
        args.learning_rate=3e-5
        require(args.target_model_backend=='sglang' and args.sglang_attention_backend=='fa3','Teacher backend changed')
        import torch
        import torch._functorch.config as compiler
        compiler.donated_buffer=False
        self.teacher=self.runtime.setup.load_target_model(args)
        self.teacher_model=self.teacher.model_runner.model.eval().requires_grad_(False)
        self.components=self.runtime.setup.load_target_embeddings(args).eval().requires_grad_(False)
        self.draft,self.model_config,_,mask=self.runtime.setup.load_draft_stack(args,self.teacher)
        parameters=list(self.draft.parameters())
        require(mask==128002 and self.draft.block_size==10 and self.draft.target_layer_ids==[1,8,15,22,29],
                'Qualified DFlash architecture changed')
        require(len(parameters)==58 and sum(p.numel() for p in parameters)==PARAMETERS and
                all(p.dtype==torch.bfloat16 and p.requires_grad for p in parameters),'Full-weight BF16 inventory changed')
        require(float(getattr(self.model_config,'attention_dropout',0.))==0. and
                not any(getattr(m,'gradient_checkpointing',False) for m in self.draft.modules()),'Draft stochastic architecture changed')
        qualified_native.CORE.verify_checkpoint(self.draft,config['released']['path'])
        self.frozen_versions=self._frozen_versions()
        from experiments.dflash_lower_lr_control.training import LowerLRTrainer,RECIPE
        group=config['original_group']
        self.trainer=LowerLRTrainer(self.draft,self.components.embed_tokens,self.components.lm_head,self.teacher_features,
            episode_id=config['episode_id'],target_id='T0',target_identity_sha256=config['target']['identity_sha256'],
            released_checkpoint_sha256=config['released']['checkpoint_identity_sha256'],mask_token_id=mask,
            attention_backend='flex_attention',check_frozen=self.check_frozen,artifact_dir=self.out/'accounting',
            retained_training=group['training'],source_references=[p['F']['kl'] for p in group['reference_pairs']['source']],
            heldout_references=[p['F']['kl'] for p in group['reference_pairs']['heldout']])
        self.completed=False
        write_json(self.out/'AUTHOR_RUNTIME.json',dict(loader_args=vars(args),recipe=RECIPE,
            schedule='Fresh released DFlash/AdamW/seed17; 0→8, four passes over four source responses, accumulation2',
            source_diagnostics='PR9 fixed source anchors; PR10 held-out anchors; eval/no_grad and RNG restored',
            donated_buffer=False,checkpoint_exports=['L8'],optimizer_state_exports=0,
            teacher_context_limit=32768,serving_input_limit=base.INPUT_LIMIT,
            author_sources=followup.check_followup_sources(config)))

    def _export_l8(self):
        import torch
        from safetensors.torch import save_file
        started=time.perf_counter()
        parent=Path(self.config['episode_out'])/'checkpoints';parent.mkdir(exist_ok=True)
        path,staging=parent/'L8',parent/'.L8.staging'
        require(not path.exists() and not staging.exists(),'Never overwrite L8 export')
        state={name:value.detach().cpu().contiguous() for name,value in self.draft.state_dict().items()}
        require(len(state)==58 and sum(v.numel() for v in state.values())==PARAMETERS and
                all(v.dtype==torch.bfloat16 and bool(torch.isfinite(v).all()) for v in state.values()),
                'L8 exported inventory or precision differs')
        required=sum(v.numel()*v.element_size() for v in state.values())+CHECKPOINT_METADATA_BYTES
        require(shutil.disk_usage(parent).free>=required+self.config.get('reserve_bytes',MIN_RESERVE_BYTES),
                'L8 export would cross storage reserve')
        staging.mkdir();save_file(state,str(staging/'model.safetensors'))
        (staging/'config.json').write_text(self.model_config.to_json_string())
        (staging/'version.txt').write_text('1')
        for entry in staging.iterdir():
            with entry.open('rb') as stream:os.fsync(stream.fileno())
        self.native.CORE.verify_checkpoint(self.draft,staging)
        files={entry.name:dict(bytes=entry.stat().st_size,sha256=sha(entry)) for entry in sorted(staging.iterdir())}
        os.rename(staging,path)
        fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return dict(path=str(path),files=files,checkpoint_identity_sha256=digest(files),checkpoint_id='L8',
            group_id=self.config['episode_id'],optimizer_steps=8,learning_rate=3e-5,export_s=time.perf_counter()-started)

    def train_group(self,body):
        require(not self.completed and body=={},'One bounded 0→8 trajectory only')
        self.completed=True
        import torch
        from experiments.dflash_lower_lr_control.loss import evaluate_responses
        from experiments.dflash_baseline_learning_curve.experiment import preserve_invalid_numbers
        group=self.config['original_group'];responses=group['support']
        started=time.perf_counter()
        result=dict(status='running',group_id=self.config['episode_id'],checkpoint=None,
            accounting=dict(status='not_started',optimizer_steps_total=0,response_presentations_total=0,batches=[]),
            losses={'source':[],'heldout':[]},loss_diagnostics={},costs={})
        population=None
        try:
            write_json(self.out/'TRAINING_BEGIN.json',result)
            for response in responses:
                saved=json.loads(base.checked_binding(response['metadata_binding']).read_text())
                require(all(response.get(k)==v for k,v in saved.items()),'Original support metadata changed')
            torch.cuda.reset_peak_memory_stats()
            result['accounting']=self.trainer.train_to_step8(responses)
            result['checkpoint']=checkpoint=self._export_l8()
            self.trainer.bind_l8_checkpoint(checkpoint['checkpoint_identity_sha256'])
            write_json(self.out/'TRAINED_L8.json',result)
            for population in ('source','heldout'):
                records=[{**r,'group_id':r['episode_id'],'request_id':r['record_id'],'role':'feedback'} for r in responses] if population=='source' else [p['F']['native'] for p in group['reference_pairs']['heldout']]
                refs=[p['F']['kl'] for p in group['reference_pairs'][population]]
                receipt=evaluate_responses(self.draft,self.components.embed_tokens,self.components.lm_head,
                    self.teacher_features,records,refs,population=population,checkpoint_id='L8',
                    checkpoint_identity_sha256=checkpoint['checkpoint_identity_sha256'],mask_token_id=128002,
                    attention_backend='flex_attention')
                result['losses'][population]=receipt['response_losses']
                result['loss_diagnostics'][population]=receipt
                self.check_frozen();self.trainer._check_owner()
                write_json(self.out/('LOSS_'+population.upper()+'.json'),receipt)
        except BaseException as error:
            if result['accounting'].get('status')=='not_started':
                result['accounting'].update(status='failed',
                    optimizer_steps_total=getattr(self.trainer,'optimizer_steps',0),
                    response_presentations_total=getattr(self.trainer,'response_presentations',0))
                for field in ('attempted_response_calls','completed_backward_presentations','uncommitted_backward_presentations'):
                    if hasattr(self.trainer,field):
                        result['accounting'][field]=getattr(self.trainer,field)
            if hasattr(error,'training_partial'):
                partial=error.training_partial
                result['accounting']={**partial,'optimizer_steps_total':partial.get('optimizer_steps',0),
                    'response_presentations_total':partial.get('response_presentations',0),
                    'batches':partial.get('completed_batches',[])}
                result['training_partial']=partial
            if hasattr(error,'diagnostic_partial'):
                result['failure_detail']=error.diagnostic_partial
                if population is not None:
                    result['losses'][population]=error.diagnostic_partial.get('response_losses',[])
            result.update(status='failed',error_type=type(error).__name__,error=str(error))
            result['costs'].update(worker_trajectory_wall_s=time.perf_counter()-started,
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())
            result=preserve_invalid_numbers(result)
            write_json(self.out/'PARTIAL.json',result);error.diagnostic_partial=result
            raise
        result.update(status='passed')
        result['costs'].update(result['accounting']['costs'])
        result['costs'].update(export_s=result['checkpoint']['export_s'],optimizer_state_save_s=0.,checkpoint_transfer_s=0.,
            worker_trajectory_wall_s=time.perf_counter()-started,
            source_loss_diagnostic_s=result['loss_diagnostics']['source']['diagnostic_wall_s'],
            heldout_loss_diagnostic_s=result['loss_diagnostics']['heldout']['diagnostic_wall_s'],
            source_diagnostic_teacher_preparation_s=sum(r['phase_seconds']['teacher_preparation_s'] for r in result['losses']['source']),
            heldout_diagnostic_teacher_preparation_s=sum(r['phase_seconds']['teacher_preparation_s'] for r in result['losses']['heldout']),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
            accounting='Training includes teacher preparation; readonly evaluation/export/loading are separate; nested timers must not be added twice.')
        write_json(self.out/'RESULT.json',result)
        return result


def worker_main():
    from http.server import BaseHTTPRequestHandler,HTTPServer
    import traceback
    config=base._worker_config();check_sources(config);base.check_dependencies(config['workspace_root'])
    allocation=os.environ.pop('CUDA_VISIBLE_DEVICES')
    try:base.check_environment(config)
    finally:os.environ['CUDA_VISIBLE_DEVICES']=allocation
    require(config['kind'] in ('sd','train'),'Only L8 serving/training worker roles')
    service=TrainerService(config) if config['kind']=='train' else ServingService(config)
    poisoned=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.0'
        def log_message(self,*_):pass
        def do_POST(self):
            try:
                require(self.headers.get('X-Two-Update-Token')==config['rpc_token'],'Foreign L8 client')
                length=int(self.headers.get('Content-Length','0'))
                require(0<length<=64*2**20,'Invalid RPC payload length')
                body,action=json.loads(self.rfile.read(length)),self.path.lstrip('/')
                require(not poisoned,'L8 worker poisoned; no automatic retry')
                if action=='ready':value=dict(status='passed',pid=os.getpid())
                else:
                    allowed={'train_group'} if config['kind']=='train' else {'generate','publish','flush'}
                    require(action in allowed,'Operation outside the explicit L8 protocol')
                    value=getattr(service,action)(body)
                status=200
            except Exception as error:
                poisoned.append(repr(error))
                value=dict(status='failed',error=repr(error),traceback=traceback.format_exc())
                if hasattr(error,'diagnostic_partial'):value['diagnostic_partial']=error.diagnostic_partial
                write_json(Path(config['worker_out'])/('FAILURE_%d.json'%len(poisoned)),value);status=500
            payload=json.dumps(value,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    server=HTTPServer(('127.0.0.1',config['port']),Handler)
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:server.serve_forever(poll_interval=.25)
    finally:server.server_close();service.close()


if __name__=='__main__':
    require(sys.argv[1:]==['--worker'],'Use the explicit lower-LR control entry point')
    worker_main()
