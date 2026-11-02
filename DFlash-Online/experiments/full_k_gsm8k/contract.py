"""The prespecified horizon, identities and numerical settings for this run."""
import hashlib
import json
from pathlib import Path

from experiments.osd_persistent.data import validate_manifest


ARMS = ("K", "K_HALF")
METHODS = ("F", "O", "K", "K_HALF")
LEARNING_RATES = {"K": 3e-4, "K_HALF": 1.5e-4}
EVENTS = 58
RESPONSES_PER_EVENT = 128
STEPS_PER_EVENT = 128
PRESENTATIONS_PER_EVENT = 256
INTERMEDIATE_EVENTS = (14, 29, 44)
NEW_BUDGET = {
    "training_events": 116,
    "optimizer_steps": 14848,
    "response_presentations": 29696,
    "response_forwards": 29696,
    "response_backwards": 29696,
    "online_generations": 14946,
    "terminal_preparation_forwards": 14848,
    "intermediate_generations": 768,
    "final_generations": 12288,
    "correctness_generations": 8,
    "total_generations": 28010,
    "disposable_optimizer_steps": 0,
}
SERVING = {
    "block_size": 16, "max_new_tokens": 2048, "context_limit": 4096,
    "concurrency": 1, "temperature": 0, "precision": "bfloat16",
    "attention_backend": "fa3", "reasoning": "off",
    "cache_policy": "idle_full_flush_before_every_request_and_publication",
}


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_plan(plan):
    """Validate the external sealed contract once, before model work."""
    if plan["schema"] != "full_k_gsm8k_v1" or plan["arms"] != list(ARMS):
        raise ValueError("Expected K then K_HALF in the fixed full-stream campaign")
    if plan["seed"] != 17 or plan["events"] != EVENTS:
        raise ValueError("Seed or full training horizon changed")
    if plan["budget"] != NEW_BUDGET or plan["serving"] != SERVING:
        raise ValueError("Scientific work or serving contract changed")
    if plan["intermediate_events"] != list(INTERMEDIATE_EVENTS):
        raise ValueError("Intermediate evaluation endpoints changed")
    if plan["learning_rates"] != LEARNING_RATES or plan["automatic_retry"] is not False:
        raise ValueError("Learning rates or no-retry policy changed")
    if plan["execution_source"]["dirty"] or len(plan["execution_source"]["commit"]) != 40:
        raise ValueError("A published clean execution commit is required")
    manifest = json.loads(Path(plan["manifest"]).read_text())
    checked = validate_manifest(manifest)
    if (manifest["policy"]["cap_policy"] != "fixed2048"
            or manifest["policy"]["train_context_limit"] != 2272
            or manifest["policy"]["serving_context_limit"] != 4096):
        raise ValueError("Original full O request preparation must remain unchanged")
    if stable_digest(manifest) != plan["manifest_binding"]:
        raise ValueError("Original training/evaluation manifest changed")
    return {"status": "passed", "manifest": checked, "budget": NEW_BUDGET}
