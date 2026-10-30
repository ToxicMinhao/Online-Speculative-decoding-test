"""Sealed scope for the unchanged two-seed O/K coding comparison."""

import hashlib
import json

SEEDS = (17, 29)
ARMS = ("O", "K")
METHODS = ("F", "O", "K")
CELLS = ((17, 2), (17, 4), (29, 2), (29, 4))
REPEATS = 3
NEW_BUDGET = {
    "feedback": 1024,
    "terminal_teacher_forwards": 1024,
    "optimizer_steps": 2048,
    "presentations": 4096,
    "evaluation": 4608,
    "generations": 5632,
    "optimizer_previews": 0,
    "qualification_model_work": 0,
}


def method_order(ordinal, repeat):
    base = ("F", "O", "K") if ordinal % 2 == 0 else ("F", "K", "O")
    shift = (ordinal + repeat) % 3
    return base[shift:] + base[:shift]


def evaluation_schedule(requests, *, seed=None, event=None):
    cells = CELLS if seed is None and event is None else ((seed, event),)
    if any(cell not in CELLS for cell in cells):
        raise ValueError("Only the four fixed coding evaluation cells are allowed")
    return [
        {"seed": current_seed, "event": current_event,
         "request_id": request["request_id"], "panel_ordinal": ordinal,
         "repeat": repeat, "method": method, "position": position}
        for current_seed, current_event in cells
        for ordinal, request in enumerate(requests)
        for repeat in range(REPEATS)
        for position, method in enumerate(method_order(ordinal, repeat))
    ]


def data_binding(training, evaluation):
    encoded = json.dumps({"training": training, "evaluation": evaluation},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def training_contract(plan, seed):
    if seed not in SEEDS:
        raise ValueError("Only the two fixed training seeds are authorized")
    return {"schema": "ok_coding_v1", "training_seed": seed,
            "data_binding": plan["data_binding"],
            "execution_commit": plan["execution_source"]["commit"],
            "events": [1, 2, 3, 4], "responses_per_event": 128,
            "passes_per_event": 2}


def validate_requests(requests, *, evaluation):
    identities, prompts = set(), set()
    for ordinal, request in enumerate(requests):
        prompt = tuple(request["prompt_ids"])
        if request["request_id"] in identities or prompt in prompts:
            raise ValueError("Repeated request identity or complete prompt")
        if (not 1 <= len(prompt) <= 224 or request["max_new_tokens"] != 2048
                or any(type(token) is not int or token < 0 for token in prompt)):
            raise ValueError("Require complete 1..224-token prompts and output cap2048")
        if request["stop_token_ids"] != [151645, 151643]:
            raise ValueError("Pinned target termination tokens changed")
        if any(key in request for key in ("response_ids", "finish_reason", "output", "solution", "answer")):
            raise ValueError("Inputs must not contain output labels")
        if request["request_index"] != ordinal:
            raise ValueError("Preserve the sealed request order")
        if evaluation and request["panel_ordinal"] != ordinal:
            raise ValueError("Preserve all evaluation panel ordinals")
        identities.add(request["request_id"])
        prompts.add(prompt)
    return identities, prompts


def validate_plan(plan):
    if plan.get("new_budget") != NEW_BUDGET:
        raise ValueError("Work budget differs from the fixed coding comparison")
    if set(plan["configs"]) != {"17", "29"}:
        raise ValueError("One stable config is required for each fixed seed")
    if ([window["event_index"] for window in plan["windows"]] != [1, 2, 3, 4]
            or any(window["event_id"] != f"event_{window['event_index']:03d}"
                   or len(window["requests"]) != 128 for window in plan["windows"])):
        raise ValueError("Require four ordered 128-response windows")
    training = [request for window in plan["windows"] for request in window["requests"]]
    evaluation = plan["evaluation_requests"]
    if len(evaluation) != 128:
        raise ValueError("Require the fixed128-request evaluation panel")
    train_ids, train_prompts = validate_requests(training, evaluation=False)
    eval_ids, eval_prompts = validate_requests(evaluation, evaluation=True)
    if train_ids & eval_ids or train_prompts & eval_prompts:
        raise ValueError("Training and evaluation overlap")
    if plan["data_binding"] != data_binding(training, evaluation):
        raise ValueError("Sealed ordered input binding differs")
    if plan["evaluation_schedule"] != evaluation_schedule(evaluation):
        raise ValueError("Sealed four-cell evaluation schedule differs")
    if (plan.get("training_seeds") != [17, 29]
            or plan.get("stop_step") != 512
            or plan.get("inherited_horizon") != 7424
            or plan.get("no_automatic_retry") is not True):
        raise ValueError("Preserve fixed seed order, horizon and four-event stop")
    if len(plan["execution_source"]["commit"]) != 40:
        raise ValueError("Require the full published execution commit")
    return {"status": "passed", "training_requests": 512, "evaluation_requests": 128,
            "training_seeds": list(SEEDS), "events": [1, 2, 3, 4],
            "new_budget": NEW_BUDGET, "data_binding": plan["data_binding"]}
