"""Select the exact retained inputs; no dataset search or model-output selection."""

from copy import deepcopy
import hashlib
import re
import unicodedata

from .contract import EVENT_DOMAINS


def normalize_prompt(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def user_text(rendered):
    """Compare user content as well as full prompts with different instructions."""
    turns = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", rendered, re.S)
    return "\n".join(turns) if turns else rendered


def _require_ids(rows, expected, name):
    if [row["request_id"] for row in rows] != expected:
        raise ValueError(f"{name}: retained request identities or order changed")


def _request(row, *, domain, experiment_id, ordinal, stop_ids):
    result = deepcopy(row)
    result.update(
        domain=domain,
        experiment_id=experiment_id,
        source_request_id=row["request_id"],
        original_source_id=row.get("source_id", row.get("dataset_id")),
        original_request_index=row["request_index"],
        request_index=ordinal,
        stop_token_ids=list(row.get("stop_token_ids", stop_ids)),
    )
    ids = result["prompt_ids"]
    if (not ids or any(type(token) is not int or token < 0 for token in ids)
            or result["max_new_tokens"] != 2048 or len(ids) + 2048 > 2272
            or result["stop_token_ids"] != list(stop_ids)):
        raise ValueError("Retained full prompt/output cap/stop binding violates the fixed recipe")
    if not result["original_source_id"]:
        raise ValueError("Original dataset identity is required")
    return result


def build_inputs(math_manifest, coding_training, coding_evaluation, *,
                 math_canonicals, coding_canonicals, decode, stop_ids):
    """Bind old identities without changing text, tokens, or replay record IDs.

    `decode` is the pinned tokenizer's CPU-only decoder. No response or dataset
    answer is used to choose requests; canonicals validate the fixed endpoint.
    Domain information is confined to this preparation/reporting layer.
    """
    math_training = math_manifest["train_requests"][:512]
    math_evaluation = math_manifest["eval_requests"][:128]
    _require_ids(math_training, [f"train_{n:05d}" for n in range(512)], "math training")
    _require_ids(math_evaluation, [f"eval_{n:05d}" for n in range(1, 129)], "math evaluation")
    _require_ids(coding_training, [f"coding_training_{n:05d}" for n in range(512)], "coding training")
    _require_ids(coding_evaluation, [f"coding_evaluation_{n:05d}" for n in range(128)], "coding evaluation")

    training, events = [], []
    cursors = {"math": 0, "coding": 0}
    sources = {"math": math_training, "coding": coding_training}
    for event, domain in enumerate(EVENT_DOMAINS, 1):
        start = cursors[domain]
        rows = []
        for row in sources[domain][start:start + 128]:
            ordinal = len(training)
            request = _request(row, domain=domain, experiment_id=f"fresh_{ordinal:04d}",
                               ordinal=ordinal, stop_ids=stop_ids)
            request.update(global_ordinal=ordinal, event=event)
            training.append(request)
            rows.append(request["request_id"])
        events.append({"event": event, "domain": domain, "record_ids": rows})
        cursors[domain] += 128

    evaluation, canonicals = [], {}
    for index in range(128):
        for domain, row, references in (
                ("math", math_evaluation[index], math_canonicals),
                ("coding", coding_evaluation[index], coding_canonicals)):
            ordinal = len(evaluation)
            request = _request(row, domain=domain, experiment_id=f"panel_{ordinal:03d}",
                               ordinal=ordinal, stop_ids=stop_ids)
            request["panel_ordinal"] = ordinal
            evaluation.append(request)
            reference = deepcopy(references[request["request_id"]])
            if (not reference["response_ids"] or reference["finish_reason"] not in ("eos", "length")
                    or not reference.get("source_record_path")):
                raise ValueError("Missing complete retained canonical output/termination provenance")
            if "prompt_ids" in reference and reference["prompt_ids"] != request["prompt_ids"]:
                raise ValueError("Canonical reference belongs to a different rendered input")
            canonicals[request["request_id"]] = reference

    # Check source identity, exact rendered IDs, normalized rendered text, and
    # normalized user text across both domains and both train/evaluation sets.
    seen = {key: {} for key in ("source", "tokens", "rendered", "user")}
    for request in training + evaluation:
        rendered = decode(request["prompt_ids"])
        identities = {
            "source": (request["domain"], str(request["original_source_id"])),
            "tokens": tuple(request["prompt_ids"]),
            "rendered": normalize_prompt(rendered),
            "user": normalize_prompt(user_text(rendered)),
        }
        for kind, value in identities.items():
            if value in seen[kind]:
                raise ValueError(f"Fixed-input duplicate ({kind}): {seen[kind][value]} / {request['request_id']}")
            seen[kind][value] = request["request_id"]
        request["normalized_rendered_sha256"] = hashlib.sha256(identities["rendered"].encode()).hexdigest()
        request["normalized_user_sha256"] = hashlib.sha256(identities["user"].encode()).hexdigest()
    return {
        "training": training, "events": events, "evaluation": evaluation, "canonicals": canonicals,
        "checks": {"training": 1024, "evaluation": 256, "duplicates": 0,
                   "selected_using_outputs": False, "new_tokenization": False},
    }
