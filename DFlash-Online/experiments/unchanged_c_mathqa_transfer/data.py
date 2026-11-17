"""Input-only MathQA reselection and the fixed N/C/S activation schedule.

PR51's normalization, source rendering and historical input matcher are reused.
Only the prospective ranking literal, request namespace and arm schedule differ.
"""
from copy import deepcopy

from experiments.mathqa_fixed_lr_transfer import data as historical

normalize = historical.normalize
digest = historical.digest
token_digest = historical.token_digest
fingerprints = historical.fingerprints
history_matches = historical.history_matches
SOURCE_REVISION = historical.SOURCE_REVISION
SOURCE_BLOB = historical.SOURCE_BLOB
FINGERPRINT_FIELDS = historical.FINGERPRINT_FIELDS
PANEL_SIZE, WARMUP_SIZE, BLOCK_SIZE = 256, 2, 64
SELECTION_PREFIX = "kplus_unchanged_c_mathqa_transfer_20261117_v1"
REQUEST_PREFIX = "mathqa-c-transfer"
METHODS = ("N", "C", "S")
PERMUTATIONS = (("N", "C", "S"), ("N", "S", "C"), ("C", "N", "S"),
                ("C", "S", "N"), ("S", "N", "C"), ("S", "C", "N"))


def rank_key(source_id, normalized_content_sha256):
    return digest(SELECTION_PREFIX + "|" + source_id + "|" + normalized_content_sha256)


def rank_eligible(rows):
    return sorted(rows, key=lambda row: (rank_key(row["source_id"], row["normalized_content_sha256"]),
                                         row["source_id"]))


def seal_requests(eligible):
    ranked = rank_eligible(eligible)
    if len(ranked) < PANEL_SIZE + WARMUP_SIZE:
        raise ValueError("Fewer than 258 audited eligible requests; no fallback panel")
    roles = {}
    for role, selected in (("scored", ranked[:256]), ("warmup", ranked[256:258])):
        rows = []
        for ordinal, original in enumerate(selected):
            row = deepcopy(original)
            row.update(role=role, panel_ordinal=ordinal, ordinal=ordinal, request_index=ordinal,
                request_id=f"{REQUEST_PREFIX}-{original['source_row']:04d}",
                original_index=original["source_row"], max_new_tokens=2048,
                stop_token_ids=[151645, 151643], collect_feedback=False, dataset="mathqa",
                selection_key=rank_key(original["source_id"], original["normalized_content_sha256"]))
            rows.append(row)
        roles[role] = rows
    return roles


def validate_request(row, role, ordinal):
    if (row["request_id"] != f"{REQUEST_PREFIX}-{row['source_row']:04d}"
            or row["selection_key"] != rank_key(row["source_id"], row["normalized_content_sha256"])):
        raise ValueError("New C-transfer identity or fixed input rank differs")
    # Reuse the exact established input checks through its own legacy namespace;
    # no mutation of the supplied row or historical validation rules occurs.
    legacy = dict(row, request_id=f"mathqa-lr-transfer-{row['source_row']:04d}",
        selection_key=historical.rank_key(row["source_id"], row["normalized_content_sha256"]))
    historical.validate_request(legacy, role, ordinal)


def validate_manifests(panel, warmups, adaptation):
    scored, warm = panel["requests"], warmups["requests"]
    if len(scored) != 256 or len(warm) != 2:
        raise ValueError("Require the complete new 256+2 input panel")
    for role, rows in (("scored", scored), ("warmup", warm)):
        for ordinal, row in enumerate(rows):
            validate_request(row, role, ordinal)
    selected = scored + warm
    for field in ("request_id", "source_id", *FINGERPRINT_FIELDS):
        if len({row[field] for row in selected}) != 258:
            raise ValueError("Duplicated scored/warmup input: " + field)
    if [row["source_id"] for row in selected] != [row["source_id"] for row in rank_eligible(selected)]:
        raise ValueError("Sealed input rank order changed")
    train = adaptation["train_requests"]
    if (adaptation["start_ordinal"] != 512 or adaptation["stop_ordinal"] != 768
            or len(train) != 256 or [r["request_id"] for r in train] != [f"train_{i:05d}" for i in range(512, 768)]
            or len({r["source_id"] for r in train}) != 256):
        raise ValueError("Original MathQA continuation stream changed")
    for ordinal, row in enumerate(train, 512):
        if (row["source_id"].split(":")[:2] != ["mathqa", "train"] or row["request_index"] != ordinal + 1
                or row["max_new_tokens"] != 2048 or row["stop_token_ids"] != [151645, 151643]
                or row["collect_feedback"] is not True or len(row["prompt_ids"]) + 2048 > 4096):
            raise ValueError("Original continuation input contract differs")
    for field in ("source_id", "prompt_ids_sha256"):
        exposed = {row[field] for row in train}
        if any(row[field] in exposed for row in selected):
            raise ValueError("Held-out inputs overlap the continuation stream")
    return dict(status="passed", scored=256, warmup=2, adaptation=256,
                new_canonical_generations=258, no_new_model_work=True)


def evaluation_schedule(requests):
    if len(requests) != 256 or [row["panel_ordinal"] for row in requests] != list(range(256)):
        raise ValueError("Require the fixed 256-request scored panel")
    result = []
    for repeat in range(3):
        for block in range(4):
            for position, method in enumerate(PERMUTATIONS[(4 * repeat + block) % 6]):
                result.append(dict(activation=len(result), repeat=repeat, block_ordinal=block,
                    position=position, method=method, request_ordinals=list(range(64 * block, 64 * (block + 1)))))
    return result
