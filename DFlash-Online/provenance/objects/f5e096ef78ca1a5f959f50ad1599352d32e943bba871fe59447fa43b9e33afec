"""One native SGLang generation → OSD update → publication qualification.

This uses two disclosed responses and one update event. It does not launch the
128-response campaign, test adaptation efficacy, or modify the selected recipe.
Run the main process on the training GPU; the isolated worker owns serving GPU 0.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from .controller import OnlineController
from .qualify import PROMPTS, assert_state_equal, check_saved_training_state
from .recipe import load_event_config
from .storage import write_json
from .training import SpecForgeEventTrainer


def check_recipe(config_path):
    """Record and check the selected recipe before applying the small event override."""
    config, recipe = load_event_config(config_path)
    actual = {
        "optimizer": recipe.optimizer,
        "learning_rate": config.training.learning_rate,
        "weight_decay": recipe.weight_decay,
        "shuffle_responses": recipe.shuffle_responses,
        "kl_precision": recipe.kl_precision,
        "attention_backend": config.training.attention_backend,
        "accumulation_steps": config.training.accumulation_steps,
        "num_anchors": config.training.num_anchors,
        "responses_per_event": recipe.responses_per_event,
        "passes_per_event": recipe.passes_per_event,
        "max_events": recipe.max_events,
        "max_steps": config.training.max_steps,
        "seed": config.training.seed,
    }
    expected = {
        "optimizer": "bf16_adamw", "learning_rate": 3e-4,
        "weight_decay": 0.01, "shuffle_responses": True,
        "kl_precision": "bfloat16", "attention_backend": "flex_attention",
        "accumulation_steps": 2, "num_anchors": 512,
        "responses_per_event": 128, "passes_per_event": 2,
        "max_events": 58, "max_steps": 7424, "seed": 17,
    }
    assert_state_equal(actual, expected, "selected_recipe")
    return actual


def admit_storage(root, draft_path):
    """Budget the first event only; no old state exists during its atomic save."""
    released_bytes = sum(path.stat().st_size for path in Path(draft_path).glob("*.safetensors"))
    if not released_bytes:
        raise ValueError("Released draft safetensors are required for storage admission")
    # BF16 parameters and two BF16 Adam moments, plus one HF export. The pending
    # state is renamed, not copied. A 5% serialization allowance and 2 GiB of
    # feedback/compiler work space supplement the required 8 GiB free reserve.
    checkpoint_bytes = (4 * released_bytes * 105 + 99) // 100
    budget = {
        "scope": "one fresh event; no second event or old checkpoint overlap",
        "released_safetensors_bytes": released_bytes,
        "state_and_export_bytes_with_allowance": checkpoint_bytes,
        "feedback_and_compile_allowance_bytes": 2 * 2**30,
        "reserve_bytes": 8 * 2**30,
        "free_bytes": shutil.disk_usage(root).free,
    }
    budget["required_bytes"] = checkpoint_bytes + 10 * 2**30
    budget["admitted"] = budget["free_bytes"] >= budget["required_bytes"]
    write_json(root / "STORAGE_ADMISSION.json", budget)
    if not budget["admitted"]:
        raise ValueError("Insufficient storage for the single update and reserve")
    return budget


def exercise_loop(controller, inputs, root, summary, *, released_checkpoint):
    """Exercise the public loop; CPU doubles test its ordering and failure gates."""
    backend = controller.backend

    for index, prompt_ids in enumerate(inputs[:2], start=1):
        record = controller.serve(
            prompt_ids, request_id=f"qualification_{index}", session_id="qualification",
            request_index=index, max_new_tokens=128,
        )
        summary["serving"].append(record)
        replay = controller.evaluate(
            prompt_ids, canonical_response_ids=record["response_ids"],
            label=f"capture_disabled_{index}", max_new_tokens=128,
        )
        summary["replays"].append(replay)
        assert_state_equal(
            replay["record"]["metadata"]["verification_calls"],
            record["metadata"]["verification_calls"], "capture_disabled_verification_calls",
        )
        write_json(root / "PROGRESS.json", summary)

    # A one-token cap finishes from prefill alone. This replay supplies no feedback.
    short = backend.generate(
        inputs[0], max_new_tokens=1, request_id="prefill_only", session_id="evaluation",
        request_index=1, purpose="evaluation", save_to=root / "prefill_only.json",
    )
    summary["prefill_only"] = short
    assert_state_equal(short["response_ids"], summary["serving"][0]["response_ids"][:1],
                       "prefill_only_output")
    assert_state_equal(short["metadata"]["verification_calls"], 0, "prefill_only_verifications")

    event = controller.update("event_1", count=2, passes=2)
    summary["event"] = event
    training = event["training"]
    assert_state_equal(training["optimizer_steps"], 2, "optimizer_steps")
    assert_state_equal(training["response_presentations"], 4, "response_presentations")
    checkpoint = training["checkpoint_path"]
    check_checkpoint_replays(controller, inputs, root, summary,
                             released_checkpoint=released_checkpoint, checkpoint=checkpoint)


def check_checkpoint_replays(controller, inputs, root, summary, *, released_checkpoint, checkpoint):
    """Qualify an already-published checkpoint without collecting or training."""
    backend = controller.backend
    summary["publication"] = backend.check_weights(checkpoint)
    write_json(root / "PROGRESS.json", summary)
    assert_state_equal(summary["publication"]["all_equal"], True, "native_publication")

    # All three executions use the same fixed input. Switching weights must
    # invalidate affected caches; no evaluation enters the feedback buffer.
    reference = backend.generate(
        inputs[2], max_new_tokens=128, request_id="updated", session_id="evaluation",
        request_index=1, purpose="evaluation", save_to=root / "updated.json",
    )
    summary["updated_reference"] = reference
    summary["frozen_activation"] = backend.publish(released_checkpoint)
    summary["frozen_weights"] = backend.check_weights(released_checkpoint)
    write_json(root / "PROGRESS.json", summary)
    assert_state_equal(summary["frozen_weights"]["all_equal"], True, "frozen_reset")
    summary["replays"].append(controller.evaluate(
        inputs[2], canonical_response_ids=reference["response_ids"],
        label="frozen", max_new_tokens=128,
    ))
    summary["reload_activation"] = backend.publish(checkpoint)
    summary["reloaded_weights"] = backend.check_weights(checkpoint)
    write_json(root / "PROGRESS.json", summary)
    assert_state_equal(summary["reloaded_weights"]["all_equal"], True, "reloaded_publication")
    reloaded = controller.evaluate(
        inputs[2], canonical_response_ids=reference["response_ids"],
        label="updated_reloaded", max_new_tokens=128,
    )
    summary["replays"].append(reloaded)
    assert_state_equal(reloaded["record"]["metadata"]["verification_calls"],
                       reference["metadata"]["verification_calls"], "reloaded_verification_calls")
    assert_state_equal(controller.buffer.pending(), [], "pending_feedback")
    summary["trainer_status"] = controller.trainer.status()
    assert_state_equal(summary["trainer_status"]["global_step"], 2, "final_optimizer_steps")
    assert_state_equal(summary["trainer_status"]["presentations"], 4, "final_presentations")
    assert_state_equal(summary["trainer_status"]["completed_events"], ["event_1"], "events")
    write_json(root / "PROGRESS.json", summary)


def publication_resume_source(source_root):
    """Use a committed first event; preserve its failed buffer/publication record."""
    source = Path(source_root).absolute()
    plan = json.loads((source / "PLAN.json").read_text())
    inputs = json.loads((source / "INPUTS.json").read_text())
    result = json.loads((source / "RESULT.json").read_text())
    inference = {"max_new_tokens": 128, "block_size": 16, "context_limit": 2048, "seed": 17,
                 "training_objective": "osd", "training_reduction": "response_mean"}
    assert_state_equal({key: plan[key] for key in inference}, inference, "retained_protocol")
    assert_state_equal(len(inputs["input_ids"]), 3, "retained_input_count")
    training_path = source / "trainer/events/event_1/RESULT.json"
    training = json.loads(training_path.read_text())
    expected = {
        "status": "completed", "event_id": "event_1", "objective": "osd",
        "passes": 2, "optimizer_steps": 2, "response_presentations": 4,
    }
    assert_state_equal({key: training[key] for key in expected}, expected, "retained_event")
    assert_state_equal(len(result["serving"]), 2, "retained_serving_count")
    assert_state_equal(len(result["replays"]), 2, "retained_capture_replay_count")
    for replay in result["replays"]:
        assert_state_equal(replay["output_equal"], True, "retained_capture_output")
        assert_state_equal(replay["learning_unchanged"], True, "retained_capture_learning")
    checkpoint = source / "trainer/events/event_1/export"
    state = source / "trainer/training_state.pt"
    assert_state_equal(str(checkpoint), training["checkpoint_path"], "retained_checkpoint_identity")
    assert_state_equal(str(state), training["full_state_path"], "retained_state_identity")
    if not state.is_file() or not checkpoint.is_dir():
        raise FileNotFoundError("Publication recovery needs the retained state and export")
    return {
        "root": source, "plan": plan, "inputs": inputs, "training": training,
        "training_result_path": training_path, "state": state, "checkpoint": checkpoint,
        "prior_status": result["status"], "prior_error": result.get("error"),
    }


def admit_publication_resume(root):
    """The trained state/export already exist; recovery creates no new model files."""
    budget = {
        "scope": "publication/replay only; existing state and export reused in place",
        "work_allowance_bytes": 2**30, "reserve_bytes": 8 * 2**30,
        "required_bytes": 9 * 2**30, "free_bytes": shutil.disk_usage(root).free,
    }
    budget["admitted"] = budget["free_bytes"] >= budget["required_bytes"]
    write_json(root / "STORAGE_ADMISSION.json", budget)
    if not budget["admitted"]:
        raise ValueError("Insufficient storage for publication recovery and reserve")
    return budget


def run(args):
    import torch
    from accelerate.utils import set_seed
    from .sglang_backend import SGLangDFlashBackend

    root = Path(args.output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    backend = trainer = None
    summary = {"status": "running", "serving": [], "replays": []}
    try:
        resume = publication_resume_source(args.resume_publication) if args.resume_publication else None
        config_path = resume["root"] / "config.yaml" if resume else Path(args.config)
        if resume:
            assert_state_equal(args.target, resume["plan"]["target"], "recovery_target")
            assert_state_equal(args.draft, resume["plan"]["draft"], "recovery_draft")
            assert_state_equal(Path(args.config).read_text(), config_path.read_text(), "recovery_config")
            summary["storage"] = admit_publication_resume(root)
            summary["prior_evidence"] = {
                "root": str(resume["root"]), "status": resume["prior_status"],
                "error": resume["prior_error"],
                "training_result": str(resume["training_result_path"]),
                "retained_optimizer_steps": 2, "retained_response_presentations": 4,
                "retained_serving_generations": 2, "retained_capture_replays": 2,
                "old_buffer_untouched": True,
            }
        else:
            summary["storage"] = admit_storage(root, args.draft)
        summary["recipe"] = check_recipe(config_path)
        shutil.copyfile(config_path, root / "config.yaml")
        prompts = [*PROMPTS[:2], PROMPTS[4]]
        write_json(root / "PLAN.json", {
            "kind": "publication-only recovery" if resume else "infrastructure qualification",
            "efficacy_experiment": False,
            "resumed_from": str(resume["root"]) if resume else None,
            "serving_generations": 0 if resume else 2,
            "capture_disabled_generations": 3 if resume else 6,
            "events": 0 if resume else 1,
            "optimizer_steps": 0 if resume else 2,
            "response_presentations": 0 if resume else 4,
            "event_override": None if resume else {"responses": 2, "passes": 2},
            "max_new_tokens": 128, "prefill_only_cap": None if resume else 1,
            "block_size": 16, "context_limit": 2048, "seed": 17,
            "training_objective": "osd", "training_reduction": "response_mean",
            "prompts": prompts, "target": args.target, "draft": args.draft,
            "worker_python": args.worker_python, "serving_gpu": args.serving_gpu,
            "training_gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        set_seed(17)
        torch.cuda.reset_peak_memory_stats()
        backend = SGLangDFlashBackend(
            args.target, args.draft, worker_python=args.worker_python,
            output_root=root / "native", serving_gpu=args.serving_gpu,
            context_limit=2048, block_size=16,
        )
        summary["model_loading_s"] = backend.loading_s
        inputs = (resume["inputs"]["input_ids"] if resume else
                  [backend.render([{"role": "user", "content": text}]) for text in prompts])
        write_json(root / "INPUTS.json", {"prompts": prompts, "input_ids": inputs})
        trainer_root = resume["root"] / "trainer" if resume else root / "trainer"
        trainer = SpecForgeEventTrainer(
            config_path, trainer_root, objective="osd",
            resume_state=resume["state"] if resume else None,
        )
        controller = OnlineController(backend, trainer, root / "loop", buffer_capacity=2)
        if resume:
            saved = torch.load(resume["state"], map_location="cpu", weights_only=False, mmap=True)
            check_saved_training_state(trainer, saved)
            summary["restored_state_equal_before_replay"] = True
            summary["recovered_activation"] = backend.publish(resume["checkpoint"])
            check_checkpoint_replays(
                controller, inputs, root, summary,
                released_checkpoint=args.draft, checkpoint=resume["checkpoint"],
            )
        else:
            exercise_loop(controller, inputs, root, summary, released_checkpoint=args.draft)

        # Qualification-only comparison: replays must leave parameters, optimizer
        # moments and RNG exactly as saved after the one training event.
        saved = torch.load(trainer.latest, map_location="cpu", weights_only=False, mmap=True)
        check_saved_training_state(trainer, saved)
        summary["evaluation_state_preservation"] = {
            "weights_equal": True, "optimizer_equal": True, "rng_equal": True,
            "counters_equal": True,
        }
        summary["status"] = "qualified"
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        summary["wall_s"] = time.perf_counter() - started
        summary["training_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["training_peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        summary["free_bytes_after"] = shutil.disk_usage(root).free
        write_json(root / "RESULT.json", summary)
        try:
            if trainer is not None:
                trainer.close()
        finally:
            if backend is not None:
                backend.close()
        print(json.dumps({"status": summary["status"], "output": str(root)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--serving-gpu", default="0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-publication", metavar="OLD_QUALIFICATION_ROOT",
                        help="Reuse its completed event; run publication and three replays only.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
