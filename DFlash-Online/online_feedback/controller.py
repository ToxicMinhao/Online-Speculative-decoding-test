"""Synchronous serve → collect → update → publish orchestration.

The caller owns prompts and conversation history. Only serving responses enter
training; evaluation compares exact token IDs without supplying feedback.
"""
import copy
from pathlib import Path
import time

from .buffer import FeedbackBuffer
from .storage import write_json


def _filename_id(value):
    if not isinstance(value, str) or not value or value in (".", "..") or Path(value).name != value:
        raise ValueError("request, event, and evaluation IDs must be single filename components")
    return value


class OnlineController:
    def __init__(self, backend, trainer, output_root, *, buffer_capacity=None):
        self.backend = backend
        self.trainer = trainer
        self.root = Path(output_root)
        recipe = getattr(trainer, "recipe", None)
        self.passes_per_event = recipe.passes_per_event if recipe else 2
        self.max_events = recipe.max_events if recipe else None
        if buffer_capacity is None:
            buffer_capacity = (recipe.responses_per_event if recipe else None) or 8
        self.buffer = FeedbackBuffer(self.root / "feedback_buffer", capacity=buffer_capacity)

    def serve(
        self, prompt_ids, *, request_id, session_id, request_index,
        max_new_tokens, collect_feedback=True,
    ):
        """Generate once, then optionally append the exact completed response."""
        request_id = _filename_id(request_id)
        started = time.perf_counter()
        record = self.backend.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            request_id=request_id,
            session_id=session_id,
            request_index=request_index,
            purpose="serving",
            save_to=self.root / "serving" / f"{request_id}.json",
        )
        generated = time.perf_counter()
        if collect_feedback:
            self.buffer.add(record)
        completed = time.perf_counter()
        write_json(self.root / "serving" / f"{request_id}.costs.json", {
            "serve_call_s": completed - started,
            "buffer_persistence_s": completed - generated,
            "generation_validation_and_recording_s": generated - started,
            "native_request_latency_s": record.get("metadata", {}).get("request_latency_s"),
            "cost_receipt_write_is_diagnostic": True,
        })
        return record

    def update(self, event_id, count=None, *, passes=None):
        """Train a configured event; explicit sizes support bounded qualification."""
        event_id = _filename_id(event_id)
        if (self.max_events is not None
                and len(self.trainer.status()["completed_events"]) >= self.max_events):
            raise ValueError("Configured online event limit reached")
        count = self.buffer.capacity if count is None else count
        passes = self.passes_per_event if passes is None else passes
        started = time.perf_counter()
        event = self.buffer.seal_event(event_id, count)
        try:
            paths = []
            preparation = []
            for record in self.buffer.records(event["record_ids"]):
                record_id = _filename_id(record["record_id"])
                path = self.root / "features" / f"{record_id}.pt"
                prepared = self.backend.prepare_feedback(record, path)
                # Keep short responses as evidence and stop this event. Resampling
                # would silently replace the response that was actually served.
                if (prepared.get("eligible") is False
                        or prepared.get("status") == "ineligible_short_response"):
                    raise ValueError("No eligible training anchors: " + record_id)
                paths.append(path)
                preparation.append(prepared)

            training = self.trainer.train_event(event_id, paths, passes=passes)
            activation = self.backend.publish(training["checkpoint_path"])
            receipt = {
                "event_id": event_id,
                "preparation": preparation,
                "training": training,
                "activation": activation,
            }
            write_json(self.root / "events" / f"{event_id}.json", receipt)
            self.buffer.commit_event(event_id, self.backend.checkpoint_id, receipt)
        except BaseException as error:
            try:
                self.buffer.fail_event(event_id, f"{type(error).__name__}: {error}")
            except BaseException as persistence_error:
                # The sealed event already blocks retries. Keep the original
                # training/publication error visible if writing its status fails.
                error.add_note(f"Could not persist failed event: {persistence_error}")
            raise

        write_json(self.root / "events" / f"{event_id}.costs.json", {
            "update_call_s": time.perf_counter() - started,
            "includes": "buffer seal, teacher replay/persistence, training/export, activation and commit",
            "cost_receipt_write_is_diagnostic": True,
        })
        return receipt

    def evaluate(self, prompt_ids, *, canonical_response_ids, label, max_new_tokens):
        """Replay a checkpoint without buffering feedback or invoking training."""
        label = _filename_id(label)
        prompt_ids = list(prompt_ids)
        canonical_response_ids = list(canonical_response_ids)
        before = copy.deepcopy(self.trainer.status()) if self.trainer is not None else None
        started = time.perf_counter()
        record = self.backend.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            request_id=label,
            session_id="evaluation",
            request_index=1,
            purpose="evaluation",
            save_to=self.root / "evaluation" / f"{label}.json",
        )
        generated = time.perf_counter()
        after = self.trainer.status() if self.trainer is not None else None
        result = {
            "label": label,
            "record": record,
            "same_input": record["prompt_ids"] == prompt_ids,
            "output_equal": record["response_ids"] == canonical_response_ids,
            "canonical_response_ids": canonical_response_ids,
            "learning_unchanged": before == after,
        }
        write_json(self.root / "evaluation" / f"{label}.comparison.json", result)
        write_json(self.root / "evaluation" / f"{label}.costs.json", {
            "evaluation_call_s": time.perf_counter() - started,
            "generation_validation_and_recording_s": generated - started,
            "native_request_latency_s": record.get("metadata", {}).get("request_latency_s"),
            "cost_receipt_write_is_diagnostic": True,
        })
        if not result["same_input"] or not result["output_equal"] or not result["learning_unchanged"]:
            raise ValueError("Invalid checkpoint replay; preserved comparison: " + label)
        return result
