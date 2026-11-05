"""Input-only manifests for the frozen K+58 depth screen.

Callers authenticate the source files and tokenizer before rendering. Selection
uses only source identities, prompt text and token lengths; historical exposure
is reported, never used to reorder the three panels.
"""

from copy import deepcopy
import hashlib

from experiments.vfd_second_pass.data import DATASET_REVISION, SUFFIX
from experiments.mathqa_transfer.data import normalize, token_digest, TOKENIZER_REVISION


STUDY_ID = "vfd_depth35_gsm8k_v1"
READOUT_SIZE = 384
GATE_SIZE = 128
FINAL_SIZE = 384
OUTPUT_CAP = 2048
TRAIN_CONTEXT = 2272
SERVING_CONTEXT = 4096
STOP_TOKEN_IDS = [151645, 151643]


def selection_key(split, canonical_id):
    """The exact UTF-8 hash serialization is part of the sealed protocol."""
    return hashlib.sha256(f"{STUDY_ID}|{split}|{canonical_id}".encode()).hexdigest()


def render_rows(questions, split, tokenizer):
    """Render official source questions with PR25's split-specific prompts.

    PR25 training uses the raw question. Its test prompt adds the retained
    reasoning instruction. Neither path consumes a source solution.
    """
    if split not in ("train", "test"):
        raise ValueError("Use the official GSM8K train or test split")
    rows = []
    for index, question in enumerate(questions):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The source must provide a nonempty question")
        message = question + (SUFFIX if split == "test" else "")
        rendered = tokenizer.apply_chat_template(
            [dict(role="user", content=message)], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        rows.append(dict(
            source_row=index, source_id=f"gsm8k:{split}:{index}",
            question=question, user_message=message, rendered_prompt=rendered,
            prompt_ids=list(ids), prompt_ids_sha256=token_digest(ids),
            prompt_tokens=len(ids),
        ))
    return rows


def _eligible(rows, split, excluded_texts=(), excluded_tokens=()):
    """Deduplicate in canonical source order before applying the study hash."""
    eligible, exclusions = [], []
    identities, texts, tokens = {}, set(), set()
    context_limit = TRAIN_CONTEXT if split == "train" else SERVING_CONTEXT
    for row in sorted(rows, key=lambda item: item["source_row"]):
        index = row["source_row"]
        identity = f"gsm8k:{split}:{index}"
        if type(index) is not int or index < 0 or row["source_id"] != identity:
            raise ValueError("Canonical split/source-row identity differs")
        question, rendered, ids = row["question"], row["rendered_prompt"], row["prompt_ids"]
        if not isinstance(question, str) or not question.strip() or not isinstance(rendered, str):
            raise ValueError("Question and rendered prompt must be source text")
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Rendered prompt IDs must be nonempty nonnegative integers")
        token_hash = token_digest(ids)
        if row.get("prompt_ids_sha256", token_hash) != token_hash:
            raise ValueError("Rendered prompt token binding differs")
        signature = (question, rendered, tuple(ids))
        if identity in identities and identities[identity] != signature:
            raise ValueError("One source identity has conflicting prompt records")
        text_keys = {normalize(question), normalize(rendered)}
        reason = None
        if identity in identities:
            reason = "duplicate_source_identity"
        elif len(ids) + OUTPUT_CAP > context_limit:
            reason = "training_context_overflow" if split == "train" else "serving_context_overflow"
        elif text_keys & set(excluded_texts) or token_hash in excluded_tokens:
            reason = "auxiliary_fit_overlap"
        elif text_keys & texts or token_hash in tokens:
            reason = "duplicate_normalized_prompt"
        identities[identity] = signature
        if reason:
            exclusions.append(dict(source_id=identity, reason=reason, prompt_tokens=len(ids)))
            continue
        texts.update(text_keys)
        tokens.add(token_hash)
        # Copy only input fields: caller-side source solution/outcome fields do
        # not become part of either the selection rule or the serving manifest.
        eligible.append(dict(
            source_row=index, source_id=identity, stable_id=identity,
            question=question, user_message=row.get("user_message", question),
            rendered_prompt=rendered, prompt_ids=list(ids), prompt_tokens=len(ids),
            prompt_ids_sha256=token_hash, selection_key=selection_key(split, identity),
        ))
    eligible.sort(key=lambda row: (row["selection_key"], row["source_id"]))
    return eligible, exclusions


def build_manifests(train_rows, test_rows, *, historical_overlap=None, source_provenance=None):
    """Return all three fixed manifests before any native generation.

    ``historical_overlap`` maps canonical source IDs to retained evidence
    locators. Missing IDs mean no match in that inventory, not globally fresh.
    ``source_provenance`` carries the caller's authenticated file/tokenizer pins.
    """
    history = historical_overlap or {}
    train, train_excluded = _eligible(train_rows, "train")
    chosen_train = train[:READOUT_SIZE + GATE_SIZE]
    if len(chosen_train) != READOUT_SIZE + GATE_SIZE:
        raise ValueError("Need 512 eligible distinct train questions; do not shrink or replace the design")
    excluded_texts = {
        normalize(row[field]) for row in chosen_train for field in ("question", "rendered_prompt")
    }
    excluded_tokens = {row["prompt_ids_sha256"] for row in chosen_train}
    test, test_excluded = _eligible(test_rows, "test", excluded_texts, excluded_tokens)
    if len(test) < FINAL_SIZE:
        raise ValueError("Need 384 eligible fit-disjoint test questions")
    selections = dict(
        READOUT=chosen_train[:READOUT_SIZE], GATE=chosen_train[READOUT_SIZE:],
        FINAL=test[:FINAL_SIZE],
    )
    manifests = {}
    for role, selected in selections.items():
        requests = []
        for ordinal, row in enumerate(selected):
            evidence = deepcopy(history.get(row["source_id"], []))
            requests.append(dict(
                row, role=role, panel_ordinal=ordinal, request_index=ordinal,
                request_id=f"vfd-depth35-{role.lower()}-{ordinal:04d}",
                dataset_id=row["source_id"], max_new_tokens=OUTPUT_CAP,
                collect_feedback=role != "FINAL", stop_token_ids=list(STOP_TOKEN_IDS),
                prior_project_use=bool(evidence), historical_overlap_evidence=evidence,
            ))
        manifests[role] = dict(
            role=role, requests=requests, count=len(requests),
            prior_project_use_count=sum(row["prior_project_use"] for row in requests),
            unmatched_in_available_history_count=sum(not row["prior_project_use"] for row in requests),
        )
    return dict(
        schema="block_depth_manifests_v1", study_id=STUDY_ID,
        dataset_revision=DATASET_REVISION, tokenizer_revision=TOKENIZER_REVISION,
        manifests=manifests, exclusions=train_excluded + test_excluded,
        eligible_counts=dict(train=len(train), test=len(test)),
        source_counts=dict(train=len(train_rows), test=len(test_rows)),
        source_provenance=deepcopy(source_provenance or {}),
        selection=dict(
            hash_serialization="vfd_depth35_gsm8k_v1|split|gsm8k:split:source_row",
            encoding="UTF-8", digest="SHA-256", order="ascending hexadecimal digest, then source ID",
            deduplication="canonical source-row order; normalized question/rendered prompt and token IDs",
            normalization="NFKC + casefold + collapsed whitespace",
            outcomes_used=False, historical_exposure_affects_selection=False,
            source_answers_used=False, fit_disjoint=True, globally_fresh_claim=False,
        ),
        preprocessing=dict(
            enable_thinking=False, add_generation_prompt=True,
            training_prompt="raw source question", evaluation_suffix=SUFFIX,
            output_cap=OUTPUT_CAP, serving_context_limit=SERVING_CONTEXT,
            training_context_limit=TRAIN_CONTEXT, truncate_prompts=False,
            include_valid_generation_cap_hits=True,
        ),
    )
