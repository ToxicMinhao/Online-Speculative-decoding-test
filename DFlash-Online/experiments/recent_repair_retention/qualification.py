"""Bounded native/disabled/zero parity and separate active-gradient fixtures.

Three disposable Adam updates do not become serving checkpoints. Active and
long-response checks perform backward only, then discard their gradients.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time
import traceback

import torch

from experiments.functional_step_geometry.native_pilot import discarded_startup
from experiments.functional_step_geometry.qualify import cleanup
from experiments.kplus_abt.qualification import endpoint, finite_endpoint
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.prefix_guard.qualify import named_gradients, persist_comparison, pin_retained_graph_runtime
from experiments.prefix_ranking.qualify import cpu_copy
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from .reference import feature_paths, prepare_batch, read
from .training import make_trainer

CASES = ("NATIVE", "DISABLED", "ZERO", "ACTIVE")
QUALIFICATION_BUDGET = dict(temporary_optimizer_steps=3,
    ordinary_response_forwards=6, ordinary_response_backwards=6,
    startup_response_forwards=6, startup_response_backwards=6,
    active_fixture_response_forwards=1, active_fixture_response_backwards=1,
    long_response_forwards=1, long_response_backwards=1, generations=0)


def first_window(trainer, paths):
    """Consume the original pass permutation before the ordinary sampler draws."""
    order = torch.randperm(len(paths)).tolist()
    chosen = [paths[index] for index in order[:2]]
    for ordinal, path in enumerate(chosen):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if raw["record_id"] != trainer.schedule["records"][ordinal]["record_id"]:
            raise ValueError("Qualification's natural first-pair order differs")
    return chosen


def native_fields(saved):
    return {key:value for key,value in saved.items() if key not in ("observations",)}


def run_step(plan, case, output):
    from specforge.training.strategies.base import StepContext
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = feature_paths(plan)
    ledger = WorkLedger(output / "WORK.jsonl", dict(response_forwards=2,response_backwards=2,optimizer_steps=1))
    receipt = dict(status="running",case=case,execution_commit=plan["execution_commit"])
    trainer, failure = None, None
    started = time.perf_counter()
    gradients = {}
    try:
        pin_retained_graph_runtime(output)
        trainer = make_trainer(plan, output / "trainer", "NATIVE" if case == "NATIVE" else "RECENT")
        trainer.capture_first_firing = False
        receipt["startup"] = discarded_startup(trainer,paths,output / "startup")
        if case != "NATIVE":
            trainer.install_retention(enabled=case != "DISABLED",coefficient=0. if case == "ZERO" else .10)
        trainer._active_event = "qualification_"+case
        (trainer.events_root / trainer._active_event).mkdir()
        trainer.bundle.model.begin_event()
        chosen = first_window(trainer,paths)
        ledger.wrap(trainer.core.strategy,"forward_loss","response_forwards")
        ledger.wrap(trainer.backend,"backward","response_backwards")
        ledger.wrap(trainer.backend.optimizer.optimizer,"step","optimizer_steps")
        original_step = trainer._native_step

        def preclip():
            gradients.update(named_gradients(trainer))
            return original_step()

        trainer._native_step = preclip
        torch.cuda.reset_peak_memory_stats()
        for ordinal,path in enumerate(chosen):
            trainer.presentation_ordinal = ordinal
            raw = torch.load(path,map_location="cpu",weights_only=True)
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(trainer._batch(raw),StepContext(
                global_step=trainer.global_step,total_steps=trainer.horizon,collect_detailed_metrics=True))
            result.materialize_metrics()
            expected = trainer.schedule["records"][ordinal]["sampled_anchors"]
            if trainer._anchor_observations != expected:
                raise ValueError("Qualification sampled anchors differ from retained schedule")
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
        saved = endpoint(trainer,gradients)
        # Preserve tensors before any gate raises, including nonfinite failures.
        save_torch(output / "ENDPOINT.pt",saved)
        receipt["finite"] = finite_endpoint(saved)
        receipt["observations"] = deepcopy(trainer.response_records)
        receipt["physical_work"] = dict(trainer.work)
        receipt["native_selection"] = [dict(
            ratio_metrics=row["ratio_metrics"],
            sum_metrics={k:v for k,v in row["sum_metrics"].items() if not k.startswith("retention_")},
        ) for row in trainer.response_records]
        write_json(output / "NATIVE_SELECTION.json",receipt["native_selection"])
        if case != "NATIVE":
            baseline_dir = Path(plan["qualification_root"]) / "NATIVE"
            reference = torch.load(baseline_dir / "ENDPOINT.pt",map_location="cpu",weights_only=False,mmap=True)
            receipt["exact_native"] = persist_comparison(output / "EXACT_NATIVE.json",saved,reference)
            receipt["exact_selections"] = persist_comparison(output / "EXACT_SELECTIONS.json",
                receipt["native_selection"],read(baseline_dir / "NATIVE_SELECTION.json"))
        require_complete_work(read_work(ledger.path,ledger.limits),ledger.limits)
        if trainer.global_step != 769 or trainer.presentations != 1538 or trainer.core.accumulation_remainder:
            raise ValueError("Disposable qualification counters differ")
        receipt.update(status="passed",peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        failure=error
        receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            cleanup(trainer.close,failure,receipt,"qualification close")
        receipt.update(wall_s=time.perf_counter()-started,work=read_work(ledger.path,ledger.limits))
        write_json(output / "RESULT.json",receipt)
    return receipt


def active_fixture(trainer, raw):
    """Synthetic mismatch at one first proposal; no ordinary objective is changed."""
    from online_feedback.recent_repair_retention import retention_chunk_terms
    from online_feedback.first_error import first_error_mask
    from experiments.functional_step_geometry.qualify import runtime_state,restore_runtime
    device = next(trainer.bundle.model.parameters()).device
    model = trainer.bundle.model
    before_rng, runtime = capture_rng(), runtime_state(model)
    trainer.backend.optimizer.zero_grad()
    started = time.perf_counter()
    try:
        batch=prepare_batch(raw,trainer.max_length,device)
        anchors,keep,hidden=model._forward_draft_blocks(batch["input_ids"],batch["hidden_states"],batch["loss_mask"],
            max_valid_anchors=raw["input_ids"].shape[1]-raw["prompt_length"]-1)
        labels,weights,_=model._block_labels_and_weights(batch["input_ids"],batch["loss_mask"],anchors,keep)
        hidden=hidden.reshape(1,anchors.shape[1],model.block_size,-1)
        logits=model.lm_head(hidden[:,:8]).to(model.loss_dtype)
        logits.retain_grad()
        labels=labels[:,:8].clone()
        weights=weights[:,:8]
        valid_blocks=(weights[0,:,1]>0).nonzero().flatten()
        if not valid_blocks.numel():
            raise ValueError("Active fixture has no eligible first proposal")
        block=int(valid_blocks[0])
        winner=int(logits[0,block,1].argmax())
        labels[0,block,1]=(winner+1)%logits.shape[-1]
        assigned=torch.zeros_like(labels,dtype=torch.bool)
        assigned[0,block,1]=True
        floors=torch.zeros_like(weights)
        floors[0,block,1]=.5
        selected=first_error_mask(logits.argmax(-1),labels,weights,model.eos_token_ids)
        numerator,stats=retention_chunk_terms(logits,labels,weights,selected,assigned,floors)
        loss=.10*numerator/((weights*selected).sum()+1e-6)
        loss.backward()
        from experiments.adaflash_arepo.qualify import finite_gradient_check
        finite_gradient_check({name: p for name, p in trainer.bundle.draft_model.named_parameters()
                               if p.requires_grad}, trainer.output_root / "active_fixture", 0)
        if not bool(torch.isfinite(logits.grad).all()) or not bool(torch.isfinite(loss)):
            raise ValueError("Active fixture has a nonfinite loss or logit gradient")
        canonical_grad=float(logits.grad[0,block,1,int(labels[0,block,1])])
        rival_grad=float(logits.grad[0,block,1,winner])
        draft_nonzero=sum(int(bool(p.grad is not None and (p.grad != 0).any())) for p in trainer.bundle.draft_model.parameters())
        if not (float(loss)>0 and canonical_grad<0 and rival_grad>0 and draft_nonzero>0):
            raise ValueError("Active hinge did not propagate the required draft gradient")
        return dict(status="passed",synthetic_only=True,response_forwards=1,response_backwards=1,
                    optimizer_steps=0,loss=float(loss),canonical_logit_gradient=canonical_grad,
                    rival_logit_gradient=rival_grad,nonzero_draft_gradient_tensors=draft_nonzero,
                    wall_s=time.perf_counter()-started)
    finally:
        trainer.backend.optimizer.zero_grad()
        restore_runtime(model,runtime)
        restore_rng(before_rng)


def run_active(plan,output):
    from specforge.training.strategies.base import StepContext
    from experiments.functional_step_geometry.qualify import scientific_state,runtime_state,restore_runtime
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    paths=feature_paths(plan)
    trainer=None
    failure=None
    receipt=dict(status="running",case="ACTIVE",startup_response_forwards=0,startup_backwards=0,
                 optimizer_steps=0,execution_commit=plan["execution_commit"])
    began=time.perf_counter()
    try:
        pin_retained_graph_runtime(output)
        trainer=make_trainer(plan,output / "trainer","RECENT")
        trainer.capture_first_firing=False
        trainer._active_event="active_fixture"
        (trainer.events_root / trainer._active_event).mkdir()
        trainer.bundle.model.begin_event()
        before=scientific_state(trainer)
        raw=torch.load(paths[0],map_location="cpu",weights_only=True)
        receipt["active_fixture"]=active_fixture(trainer,raw)
        receipt["active_restored"]=persist_comparison(output / "ACTIVE_RESTORED.json",scientific_state(trainer),before)
        # Select by retained input length, never by loss or intervention outcome.
        lengths=[]
        for path in paths:
            item=torch.load(path,map_location="cpu",weights_only=True)
            lengths.append((item["input_ids"].shape[1],str(path)))
        _,longest=max(lengths)
        raw=torch.load(longest,map_location="cpu",weights_only=True)
        runtime=runtime_state(trainer.bundle.model)
        rng=capture_rng()
        trainer.install_retention()
        # Consume the same permutation and preceding sampler draws, without any
        # extra model forward, so the longest response uses its actual bindings.
        order=torch.randperm(len(paths)).tolist()
        draw_count=0
        for ordinal,index in enumerate(order):
            if str(paths[index]) == longest:
                trainer.presentation_ordinal=ordinal
                break
            prior=torch.load(paths[index],map_location="cpu",weights_only=True)
            device=next(trainer.bundle.model.parameters()).device
            batch=trainer._batch(prior).tensors
            trainer._original_sampler(batch["input_ids"].shape[-1],batch["loss_mask"].to(device),device,
                max_valid_anchors=prior["input_ids"].shape[1]-prior["prompt_length"]-1)
            draw_count+=1
        else:
            raise ValueError("Longest retained response absent from native first pass")
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        try:
            result=trainer.strategy.forward_loss(trainer._batch(raw),StepContext(
                global_step=768,total_steps=trainer.horizon,collect_detailed_metrics=True))
            trainer.backend.backward(result.loss,is_boundary=True)
            torch.cuda.synchronize()
            receipt["long_response"]=dict(record_id=raw["record_id"],sequence_tokens=raw["input_ids"].shape[1],
                response_forwards=1,response_backwards=1,optimizer_steps=0,wall_s=time.perf_counter()-start,
                auxiliary_extension_enabled=True,preceding_sampler_only_draws=draw_count,
                actual_presentation_id=trainer.presentation_ordinal,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        finally:
            trainer.backend.optimizer.zero_grad()
            restore_runtime(trainer.bundle.model,runtime)
            restore_rng(rng)
            trainer.presentation_ordinal=None
        receipt["long_restored"]=persist_comparison(output / "LONG_RESTORED.json",scientific_state(trainer),before)
        receipt["status"]="passed"
    except BaseException as error:
        failure=error
        receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            cleanup(trainer.close,failure,receipt,"fixture close")
        receipt["wall_s"]=time.perf_counter()-began
        write_json(output / "RESULT.json",receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--case",choices=CASES,required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    plan=read(args.plan)
    if args.case=="ACTIVE":
        run_active(plan,args.output)
    else:
        run_step(plan,args.case,args.output)
