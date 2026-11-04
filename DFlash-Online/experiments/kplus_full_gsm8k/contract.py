"""Fixed one-arm K+ training budget and retained-K endpoint contract."""
import json
from pathlib import Path

from experiments.full_k_gsm8k.contract import SERVING, stable_digest
from experiments.osd_persistent.data import validate_manifest

SCHEMA = "kplus_full_gsm8k_v1"
ARMS = ("KPLUS",)
METHODS = ("K", "KPLUS")
EVENTS = 58
RESPONSES_PER_EVENT = 128
STEPS_PER_EVENT = 128
PRESENTATIONS_PER_EVENT = 256
LEARNING_RATES = {"KPLUS": 3e-4}
NEW_BUDGET = {
    "training_events": 58, "optimizer_steps": 7424,
    "response_presentations": 14848, "response_forwards": 14848,
    "response_backwards": 14848, "online_generations": 7473,
    "terminal_preparation_forwards": 7424, "intermediate_generations": 0,
    "final_generations": 6144, "correctness_generations": 4,
    "total_generations": 13621, "disposable_optimizer_steps": 0,
    "baseline_optimizer_steps": 0,
}
HISTORICAL_K_TOTALS = {"verification_calls": 43387, "output_tokens": 314847}


def validate_plan(plan):
    """Admit the complete sealed protocol before constructing a model."""
    fixed = dict(schema=SCHEMA, arms=list(ARMS), seed=17, events=EVENTS,
                 budget=NEW_BUDGET, serving=SERVING, learning_rates=LEARNING_RATES,
                 intermediate_events=[], automatic_retry=False)
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError("K+ requires the fixed one-arm full-horizon protocol")
    execution = plan["execution_source"]
    if execution["dirty"] or len(execution["commit"]) != 40:
        raise ValueError("Publish and seal a clean execution commit before production")
    if set(plan["configs"]) != set(ARMS) or set(plan["references"]) != {"K"}:
        raise ValueError("Only KPLUS trains; retained K58 is a serving reference")
    if plan["references"]["K"] == plan["draft"]:
        raise ValueError("The retained K58 reference cannot be the released initialization")
    manifest = json.loads(Path(plan["manifest"]).read_text())
    checked = validate_manifest(manifest)
    if (len(manifest["train_requests"]) != 7473
            or manifest["policy"]["cap_policy"] != "fixed2048"
            or manifest["policy"]["train_context_limit"] != 2272
            or manifest["policy"]["serving_context_limit"] != 4096
            or stable_digest(manifest) != plan["manifest_binding"]):
        raise ValueError("The original full GSM8K stream must remain exact")
    from .evaluation import final_schedule, validate_inputs
    if plan["evaluation_schedule"] != final_schedule(plan["evaluation_requests"]):
        raise ValueError("The sealed endpoint order differs")
    canonicals = json.loads(Path(plan["canonicals"]).read_text())
    baseline = json.loads(Path(plan["baseline_counts"]).read_text())
    final_export = str(Path(plan["root"]) / "campaign_001/arms/KPLUS/trainer/events/event_058/export")
    validate_inputs(plan["evaluation_requests"], {"K": plan["references"]["K"], "KPLUS": final_export},
                    canonicals, baseline)
    return dict(status="passed", manifest=checked, budget=NEW_BUDGET)
