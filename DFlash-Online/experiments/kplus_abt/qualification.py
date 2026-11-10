"""Bounded native ABT qualification. Disposable updates never initialize training."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time
import traceback

import torch

from experiments.full_k_gsm8k.training import live_state, _finite
from experiments.functional_step_geometry.native_pilot import discarded_startup
from experiments.functional_step_geometry.qualify import cleanup, runtime_state, restore_runtime, export_disposable_state
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_coding.storage import identity
from experiments.ok_continuation.resume import differences
from experiments.prefix_guard.qualify import pin_retained_graph_runtime, persist_comparison
from experiments.prefix_ranking.qualify import cpu_copy
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.ranking import fixed_anchors
from online_feedback.storage import save_torch, write_json
from .contract import ARMS, DATASETS, QUALIFICATION_CASES
from .data import read
from .training import AdversarialBlockTrainer

NORM_RTOL, NORM_ATOL = 1e-5, 1e-8


def callback_ledger(ledger):
    """Use the same durable accounting for the two internally sequenced views."""
    from experiments.ok_coding.campaign import append_json
    def callback(kind, view, phase):
        name = {'forward':'response_forwards', 'backward':'response_backwards'}[kind]
        if phase == 'started':
            if ledger.attempts[name] >= ledger.limits[name]:
                raise RuntimeError('Qualification view budget exceeded')
            ordinal = ledger.attempts[name]+1
            append_json(ledger.path, dict(kind=name, phase=phase, ordinal=ordinal, view=view))
            ledger.attempts[name] = ordinal
        else:
            ordinal = ledger.attempts[name]
            append_json(ledger.path, dict(kind=name, phase=phase, ordinal=ordinal, view=view))
            ledger.completed[name] += 1
    return callback


def perturbation_checks(snapshots, case):
    """Declared FP32 nominal norm check; BF16 realization is measured separately."""
    from online_feedback.first_error import first_error_mask, valid_prefix_mask
    from online_feedback.kplus import first_error_groups
    rows = []
    for saved in snapshots:
        labels, weights = saved['labels'], saved['weights']
        selection = first_error_mask(saved['clean_proposals'], labels, weights, saved['eos_token_ids'])
        groups = first_error_groups(saved['clean_proposals'], labels, weights, selection, saved['eos_token_ids'])
        if (not torch.equal(selection, saved['selection']) or
            not torch.equal(saved['eligible'], valid_prefix_mask(labels, weights, saved['eos_token_ids'])) or
            not torch.equal(groups[0], saved['priority1']) or not torch.equal(groups[1], saved['priority2'])):
            raise ValueError('Captured selection, priority or perturbation scope differs')
        if case in ('ADV', 'RANDOM'):
            active = ~(saved['zero_gradient_blocks'] | saved['no_eligible_blocks'])
            intended = saved['intended_norm']
            expected = torch.where(active, saved['nominal_radius'], 0.)
            if not torch.allclose(intended, expected, rtol=NORM_RTOL, atol=NORM_ATOL):
                raise ValueError('FP32 intended perturbation norm misses its declared radius')
            if not bool(active.any()) or not bool((saved['changed_slots'][active] > 0).any()):
                raise ValueError('No nonzero gradient-directed perturbation reaches native BF16 input')
            row = dict(eligible_slots=int(saved['eligible'].sum()), active_blocks=int(active.sum()),
                changed_slots=int(saved['changed_slots'].sum()),
                nominal_radius=saved['nominal_radius'].tolist(),
                intended_norm=intended.tolist(), realized_norm=saved['realized_norm'].tolist(),
                gradient_norm=saved['gradient_norm'].tolist())
        else:
            row = dict(eligible_slots=int(saved['eligible'].sum()), clean_second_view=True)
        rows.append(row)
    return dict(status='passed', rows=rows, norm_rtol=NORM_RTOL, norm_atol=NORM_ATOL,
                perturbed_loss_increase_required=False)


class NativeInputAudit:
    """Check actual draft inputs and optimizer versions across each paired view."""
    def __init__(self, trainer):
        self.trainer, self.pending, self.pairs = trainer, None, []
        self.handle = trainer.bundle.model.draft_model.register_forward_pre_hook(self.observe, with_kwargs=True)

    @staticmethod
    def mask_parts(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key:NativeInputAudit.mask_parts(part) for key,part in value.items()}
        if hasattr(value, 'to_dense'):
            return dict(block_topology=value.to_dense().detach().cpu().clone(),
                        block_size=value.BLOCK_SIZE, sequence_lengths=getattr(value,'seq_lengths',None))
        names = ('kv_num_blocks','kv_indices','full_kv_num_blocks','full_kv_indices',
                 'q_num_blocks','q_indices','full_q_num_blocks','full_q_indices','BLOCK_SIZE','seq_lengths')
        parts = {key:NativeInputAudit.mask_parts(getattr(value,key))
                if isinstance(getattr(value,key), (torch.Tensor,dict)) else deepcopy(getattr(value,key))
                for key in names if hasattr(value,key)}
        if not parts:
            raise TypeError('Unsupported native attention-mask representation')
        return parts

    def observe(self, module, args, kwargs):
        runtime = getattr(self.trainer.bundle.model,'adversarial_block',None)
        if runtime is None:
            return
        actual = dict(context=kwargs['target_hidden'].detach().cpu().clone(),
            positions=kwargs['position_ids'].detach().cpu().clone(),
            attention=self.mask_parts(kwargs['attention_mask']),
            parameter_versions=[(name,p._version) for name,p in module.named_parameters()],
            adam_counters=[float(state['step']) for state in self.trainer.backend.optimizer.optimizer.state.values()],
            adam_tensor_versions=[[(name,value.data_ptr(),value._version) for name,value in state.items()
                                   if isinstance(value,torch.Tensor)]
                                  for state in self.trainer.backend.optimizer.optimizer.state.values()],
            rng=capture_rng())
        noise=kwargs['noise_embedding'].detach().reshape_as(runtime.embeddings)
        if runtime.view=='clean':
            if not torch.equal(noise,runtime.embeddings):
                raise ValueError('Zero delta changed native clean embeddings')
            self.pending=actual
        else:
            changed=(noise.float()-runtime.embeddings.float()).ne(0).any(-1)
            if bool((changed & ~runtime.eligible).any()):
                raise ValueError('Seed or ineligible slot was perturbed')
            if differences(actual,self.pending):
                raise ValueError('Teacher context, position, attention, stochastic state or Adam changed between views')
            self.pairs.append(dict(native_conditioning_exact=True,weights_and_moments_fixed=True,
                                   eligible_only=True,changed_slots=int(changed.sum())))
            self.pending=None

    def close(self):
        self.handle.remove()
        self.pending=None


def endpoint(trainer, gradients):
    """Comparable scientific fields only; arm/run metadata is deliberately separate."""
    return dict(weights={n:p.detach().cpu().clone() for n,p in trainer.bundle.draft_model.state_dict().items()},
        optimizer=cpu_copy(trainer.backend.optimizer.state_dict()), gradients=gradients,
        rng=capture_rng(), global_step=trainer.global_step, presentations=trainer.presentations,
        micro=trainer.core._micro, remainder=trainer.core.accumulation_remainder,
        anchors=[r['sampled_anchors'] for r in trainer.response_records],
        response_losses=[r['loss'] for r in trainer.response_records])


def run_case(root, dataset, case):
    from specforge.training.strategies.base import StepContext
    root = Path(root); plan = read(root/'sealed/PLAN.json')
    if dataset not in DATASETS or case not in QUALIFICATION_CASES:
        raise ValueError('Unknown disposable qualification case')
    output = root/'qualification'/dataset/case
    output.mkdir(parents=True, exist_ok=False)
    preparation = read(plan['shared_feedback_preparation'])
    features = [Path(r['feature_path']) for r in preparation['datasets'][dataset]]
    if len(features) != 2:
        raise ValueError('Qualification uses exactly the two declared shared feedback packets')
    arm = case if case in ARMS else 'ADV'
    views = 4 if case in ('CLEAN2','RANDOM','ADV') else 2
    ledger = WorkLedger(output/'WORK.jsonl', dict(response_forwards=views, response_backwards=views, optimizer_steps=1))
    item = plan['datasets'][dataset]
    receipt = dict(status='running', dataset=dataset, case=case, automatic_retry=False,
                   execution_commit=plan['execution_source']['commit'], binding=plan['binding'])
    trainer = None; failure = None; started = time.perf_counter()
    captures, gradients = [], {}
    input_audit = None
    try:
        pin_retained_graph_runtime(output)
        trainer = AdversarialBlockTrainer(item['configs'][arm], output/'trainer', arm=arm,
            dataset=dataset, binding=plan['binding'], original_config_text=Path(plan['original_config']).read_text(),
            origin_state=item['origin_state'], origin_export=item['origin_export'])
        receipt['complete_state'] = read(trainer.output_root/'RESUME_CHECK.json')
        receipt['startup'] = discarded_startup(trainer, features, output/'startup')
        trainer._active_event = 'disposable_'+case
        (trainer.events_root/trainer._active_event).mkdir()
        trainer.bundle.model.begin_event()
        def observe(batch, result, snapshot, row):
            snapshot['eos_token_ids'] = tuple(trainer.bundle.model.eos_token_ids)
            captures.append(snapshot)
        installed = trainer.install_two_views(enabled=case!='DISABLED',
            rho=0. if case=='ADV_ZERO' else .01, work_callback=callback_ledger(ledger),
            response_observer=observe)
        receipt['two_view_installed'] = installed
        if installed:
            input_audit = NativeInputAudit(trainer)
        if not installed:
            ledger.wrap(trainer.core.strategy, 'forward_loss', 'response_forwards')
            ledger.wrap(trainer.backend, 'backward', 'response_backwards')
        native_step = trainer.backend.optimizer.optimizer.step
        def step(*args, **kwargs):
            for name, parameter in trainer.bundle.draft_model.named_parameters():
                gradients[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            return ledger.call('optimizer_steps', native_step, *args, **kwargs)
        trainer.backend.optimizer.optimizer.step = step
        torch.cuda.reset_peak_memory_stats()
        for feature in features:
            raw = torch.load(feature, map_location='cpu', weights_only=True)
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(trainer._batch(raw), StepContext(global_step=trainer.global_step,
                total_steps=trainer.horizon, collect_detailed_metrics=True))
            result.materialize_metrics()
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
        require_complete_work(read_work(ledger.path, ledger.limits), ledger.limits)
        if (trainer.global_step != 513 or trainer.presentations != 1026 or trainer.core.accumulation_remainder
                or len(trainer.window_records)!=1 or len(trainer.response_records)!=2):
            raise ValueError('Disposable two-response/one-Adam accounting differs')
        # Save before comparisons: a failed numerical control keeps its tensors.
        saved = endpoint(trainer, gradients)
        for field in ('weights','optimizer','gradients'):
            _finite(saved[field],field)
        save_torch(output/'ENDPOINT.pt', saved)
        receipt['endpoint_identity'] = identity(output/'ENDPOINT.pt')
        if case != 'KPLUS':
            reference = torch.load(root/'qualification'/dataset/'KPLUS/ENDPOINT.pt',map_location='cpu',weights_only=False,mmap=True)
            if case in ('DISABLED','ADV_ZERO'):
                receipt['neutral_exact'] = persist_comparison(output/'NATIVE_EXACT.json', saved, reference)
            else:
                exact_fields = ('rng','anchors','global_step','presentations','micro','remainder')
                receipt['sampling_state_exact'] = persist_comparison(output/'SAMPLING_STATE_EXACT.json',
                    {key:saved[key] for key in exact_fields}, {key:reference[key] for key in exact_fields})
                # CLEAN2 may differ under actual two-view BF16 accumulation.
                from experiments.prefix_guard.qualify import compare_states
                receipt['native_difference'] = compare_states(saved, reference)
                write_json(output/'NATIVE_DIFFERENCE.json', receipt['native_difference'])
            del reference
        if installed:
            if len(input_audit.pairs) != 2 or input_audit.pending is not None:
                raise ValueError('Expected two complete native paired-input checks')
            receipt['native_input_pairs'] = input_audit.pairs
            input_audit.close(); input_audit = None
            receipt['perturbation'] = perturbation_checks(captures, case)
            save_torch(output/'VIEW_SNAPSHOTS.pt', captures)
        receipt['views'] = trainer.view_records
        receipt['observations'] = deepcopy(trainer.response_records)
        receipt['windows'] = deepcopy(trainer.window_records)
        receipt['physical_work'] = dict(trainer.work)
        if case in ARMS:
            payload = live_state(trainer)
            temporary = output/'EXPORT_INPUT.pt'
            save_torch(temporary, payload)
            receipt['export'] = str(output/'export')
            receipt['export_check'] = export_disposable_state(trainer, temporary, output/'export', payload)
            temporary.unlink()
        del saved
        if case in ARMS:
            del payload
        gradients.clear()
        gc.collect()
        if case == 'ADV':
            torch.cuda.empty_cache()
            from .resource import run as qualify_resource
            receipt['resource'] = qualify_resource(trainer, features, output/'resource')
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        receipt.update(wall_s=time.perf_counter()-started,work=read_work(ledger.path,ledger.limits),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        cleanup(lambda:write_json(output/'RESULT.json',receipt), failure, receipt, 'qualification receipt')
        if input_audit is not None:
            cleanup(input_audit.close, failure, receipt, 'input audit close')
        if trainer is not None:
            cleanup(trainer.close, failure, receipt, 'disposable trainer close')
    return receipt


def serving_case(root, dataset, case):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    root=Path(root); plan=read(root/'sealed/PLAN.json'); output=root/'qualification'/dataset/case
    receipt=read(output/'RESULT.json')
    if case not in ARMS or receipt['status']!='completed':
        raise ValueError('Only completed declared qualification exports are served')
    prepared=read(plan['shared_feedback_preparation'])['datasets'][dataset][0]
    canonical=read(prepared['record_path']); backend=None; failure=None
    ledger=WorkLedger(output/'SERVING_WORK.jsonl',dict(serving_generations=1))
    result=dict(status='running')
    try:
        backend=SGLangDFlashBackend(plan['target'],receipt['export'],worker_python=plan['worker_python'],
            output_root=output/'native',serving_gpu='0',context_limit=4096,block_size=16)
        record=ledger.call('serving_generations',backend.generate,canonical['prompt_ids'],max_new_tokens=2048,
            request_id=f'qualification_{dataset}_{case}',session_id='abt_qualification',request_index=0,
            purpose='evaluation',save_to=output/'SERVING_RAW.json')
        if record['response_ids']!=canonical['response_ids'] or record['finish_reason']!=canonical['finish_reason']:
            raise ValueError('Disposable trained export changes canonical output/termination')
        result.update(status='passed',exact_output_and_termination=True,loading_s=backend.loading_s,
            verification_calls=record['metadata']['verification_calls'],perturbation_in_serving=False)
    except BaseException as error:
        failure=error; result.update(status='failed',error=repr(error)); raise
    finally:
        if backend is not None:
            cleanup(backend.close,failure,result,'serving close')
        result['work']=read_work(ledger.path,ledger.limits)
        cleanup(lambda:write_json(output/'SERVING_CHECK.json',result),failure,result,'serving receipt')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True); parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--case',choices=QUALIFICATION_CASES,required=True)
    parser.add_argument('--serve',action='store_true')
    args=parser.parse_args()
    (serving_case if args.serve else run_case)(args.root,args.dataset,args.case)
