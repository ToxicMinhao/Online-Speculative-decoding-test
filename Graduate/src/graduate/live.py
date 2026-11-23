"""Collect real requests into the feedback format already consumed by training."""

from datetime import datetime, timezone
import time

from transformers import AutoTokenizer

from graduate.batches import load_feedback, read_feature_layout
from graduate.collection import read_requests
from graduate.feedback import ResponseRecord
from graduate.generation import load_generator
from graduate.storage import write_json


def collect_feedback(config, debug_responses=None):
    requests = read_requests(config.collection.requests)
    if debug_responses is not None:
        if not 0 < debug_responses <= len(requests):
            raise ValueError("Debug request count must fit the request list.")
        requests = requests[:debug_responses]
    layout = read_feature_layout(config)
    tokenizer = AutoTokenizer.from_pretrained(config.models.target, local_files_only=True)
    # Tokenize once. Neither feedback preparation nor training retokenizes answers.
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": request.prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False,
    ) for request in requests]
    for prompt in prompts:
        if len(prompt) + config.serving.max_new_tokens > config.serving.context_limit:
            raise ValueError("Prompt plus output allowance exceeds the context; do not truncate.")

    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.collection.output_dir / session
    run_dir.mkdir(parents=True)
    write_json(run_dir / "inputs.json", {
        "configuration": config.model_dump(mode="json"),
        "requests": [dict(request.model_dump(), prompt_ids=prompt)
                     for request, prompt in zip(requests, prompts)],
        "debug_requests": debug_responses,
    })
    started = time.perf_counter()
    with load_generator(config, run_dir / "native") as generator:
        loading_seconds = time.perf_counter() - started
        stop_ids = generator.stop_token_ids
        manifest, outcomes = [], []
        # The stable manifest points only to this collection. Failed/short responses
        # remain in the run directory; they never silently reuse an older answer.
        if any(request.role == "feedback" for request in requests):
            write_json(config.feedback.manifest, manifest)
        for index, (request, prompt) in enumerate(zip(requests, prompts)):
            capture = request.role == "feedback"
            request_dir = run_dir / f"{index:04d}"
            request_dir.mkdir()
            started = time.perf_counter()
            generated = generator.generate(
                prompt, config.serving.max_new_tokens, stop_ids,
                config.serving.block_size, capture=capture,
            )
            collection_seconds = time.perf_counter() - started
            # Save the native observations before training-packet validation.
            write_json(request_dir / "generation.json", {
                "request_id": request.record_id, "purpose": request.role,
                "prompt_ids": prompt, "response_ids": generated.response_ids,
                "finish_reason": generated.finish_reason, "trace": generated.trace,
                "runtime": generator.runtime, "collection_seconds": collection_seconds,
            })
            eligible = capture and len(generated.response_ids) > 1
            preparation_started = time.perf_counter()
            if capture:
                record = ResponseRecord(
                    record_id=f"{session}:{request.record_id}", session_id=session,
                    request_index=index, checkpoint_id=str(config.models.draft),
                    purpose="serving", prompt_ids=prompt, response_ids=generated.response_ids,
                    finish_reason=generated.finish_reason, stop_token_ids=stop_ids,
                    max_new_tokens=config.serving.max_new_tokens,
                    metadata={"request_id": request.record_id, "runtime": generator.runtime},
                )
                response_path, features_path = request_dir / "response.json", request_dir / "features.pt"
                write_json(response_path, record.model_dump(mode="json"))
                generated.capture.save(record, features_path)
                if eligible:
                    load_feedback(record, features_path, layout)
                    manifest.append({"response": str(response_path), "features": str(features_path)})
                    write_json(config.feedback.manifest, manifest)
            outcomes.append({
                "request_id": request.record_id, "role": request.role,
                "output_tokens": len(generated.response_ids), "finish_reason": generated.finish_reason,
                "verification_calls": len(generated.trace),
                "accepted_draft_tokens": sum(row["accepted_draft_tokens"] for row in generated.trace),
                "training_eligible": eligible,
                "status": "saved" if eligible else ("evaluation_only" if not capture else "no_training_anchors"),
                "prefill_forwards": 1, "verification_forwards": len(generated.trace),
                "terminal_feature_forwards": int(capture),
                "terminal_feature_input_rows": len(prompt) + len(generated.response_ids) if capture else 0,
                "feedback_preparation_seconds": time.perf_counter() - preparation_started,
                "collection_seconds": collection_seconds,
            })
            write_json(run_dir / "outcomes.json", outcomes)
            write_json(run_dir / "manifest.json", manifest)
            print(f"Saved {request.record_id}: {len(generated.response_ids)} tokens, {outcomes[-1]['status']}", flush=True)

        result = {
            "status": "collection_completed", "requests": outcomes,
            "training_responses": len(manifest), "feedback_manifest": str(config.feedback.manifest),
            "output_directory": str(run_dir), "loading_seconds": loading_seconds,
            "peak_cuda_bytes": generator.peak_cuda_bytes, "optimizer_steps": 0,
            "timing_scope": "local collection including feature capture; not an A800 serving benchmark",
        }
        write_json(run_dir / "result.json", result)
    return result
