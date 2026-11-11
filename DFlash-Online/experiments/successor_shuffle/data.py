"""Input-only held-out successor-shuffle selection, sealed before reference lookup."""

import hashlib
from collections import defaultdict
import json
from pathlib import Path

from experiments.survival_confirmation import data as previous
from experiments.survival_confirmation.data import (
    digest,
    identity,
    read,
    render_question,
    REVISION,
    SOURCE_HASHES,
    STOP_IDS,
)
from experiments.survival_head.data import feedback_requests, verify_shared_packets

PREFIX = "kplus-successor-control-v1|gsm8k|test|"
MATCHED_PREFIX = "kplus-successor-control-v1|matched-start|gsm8k|test|"
ROOT = Path("/home/minhao2/EAGLE_SPEC/research")


def row_key(index, prefix=PREFIX):
    return hashlib.sha256((prefix + str(index)).encode("utf-8")).hexdigest(), index


def select_rows(pr38_panel, *, additional_excluded_rows=(), reserved_test_rows=()):
    previous.verify_panel(pr38_panel)
    if pr38_panel["count"] != 768 or pr38_panel["selection"]["eligible_count"] != 1063:
        raise ValueError("Require the authenticated PR38 768 of1063 source panel")
    source = set(pr38_panel["selection"]["eligible_rows_in_hash_order"])
    measured = {r["source_row"] for r in pr38_panel["requests"]}
    additions = set(additional_excluded_rows) | set(reserved_test_rows)
    if any(type(i) is not int or not 0 <= i < 1319 for i in additions):
        raise ValueError("Additional exclusions require official GSM8K test rows")
    eligible = sorted(source - measured - additions, key=row_key)
    n = min(256, len(eligible) // 8 * 8)
    if n < 64:
        raise ValueError(
            "Insufficient eligible rows for prescribed64 matched-start inputs"
        )
    selected = eligible[:n]
    return selected, dict(
        pr38_eligible_count=len(source),
        pr38_measured_count=len(measured),
        pr38_panel_binding=pr38_panel["binding"],
        remaining_before_additional=len(source - measured),
        eligible_count=len(eligible),
        eligible_rows_in_hash_order=eligible,
        additional_excluded_rows=sorted(set(additional_excluded_rows)),
        reserved_test_rows=sorted(set(reserved_test_rows)),
        count=n,
        selection_prefix=PREFIX,
        matched_selection_prefix=MATCHED_PREFIX,
        outcomes_used=False,
        reference_availability_used=False,
        prior_exposure_used=False,
        policy="ascending SHA256 UTF8(prefix + official decimal row), official row tie-break; complete eight-request blocks",
    )


def matched_subset(requests):
    if len(requests) < 64:
        raise ValueError("Require64 distinct matched-start requests")
    selected = sorted(requests, key=lambda r: row_key(r["source_row"], MATCHED_PREFIX))[
        :64
    ]
    if len({r["request_id"] for r in selected}) != 64:
        raise ValueError("Duplicate matched-start request")
    return [
        dict(
            r,
            matched_start_ordinal=i,
            matched_selection_sha256=row_key(r["source_row"], MATCHED_PREFIX)[0],
        )
        for i, r in enumerate(selected)
    ]


def verify_panel(panel):
    if digest({k: v for k, v in panel.items() if k != "binding"}) != panel.get(
        "binding"
    ):
        raise ValueError("Successor panel binding changed")
    n = panel["count"]
    rows = panel["requests"]
    audit = panel["selection"]
    if type(n) is not int or not 64 <= n <= 256 or n % 8 or len(rows) != n:
        raise ValueError("Invalid sealed successor panel size")
    if [r["source_row"] for r in rows] != audit["eligible_rows_in_hash_order"][:n]:
        raise ValueError("Changed official input hash ordering")
    if any(
        audit[k]
        for k in ("outcomes_used", "reference_availability_used", "prior_exposure_used")
    ):
        raise ValueError("Selection is not input-only")
    if (
        len({r["source_row"] for r in rows}) != n
        or len({r["request_id"] for r in rows}) != n
    ):
        raise ValueError("Duplicate selected identity")
    for ordinal, r in enumerate(rows):
        if (
            r["panel_ordinal"] != ordinal
            or r["source_id"] != f"gsm8k:main:test:{r['source_row']}"
            or r["source_revision"] != REVISION
            or r["prompt_ids_sha256"] != digest(r["prompt_ids"])
            or r["selection_sha256"] != row_key(r["source_row"])[0]
            or r["stop_token_ids"] != STOP_IDS
            or r["max_new_tokens"] != 2048
            or r["collect_feedback"]
        ):
            raise ValueError("Changed successor request protocol")
    if panel["matched_start_request_ids"] != [
        r["request_id"] for r in matched_subset(rows)
    ]:
        raise ValueError("Changed matched-start selection")
    return dict(status="passed", count=n, binding=panel["binding"])


def prepare(
    pr38_panel,
    source_manifest,
    tokenizer_path,
    output,
    *,
    additional_excluded_rows=(),
    reserved_test_rows=(),
):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from experiments.mathqa_transfer.data import TOKENIZER_HASHES, normalize

    old = read(pr38_panel)
    selected, audit = select_rows(
        old,
        additional_excluded_rows=additional_excluded_rows,
        reserved_test_rows=reserved_test_rows,
    )
    source = read(source_manifest)["sources"]["gsm8k_test"]
    bound = identity(source["local_path"])
    if bound["sha256"] != SOURCE_HASHES["test"] or source["revision"] != REVISION:
        raise ValueError("Changed official source")
    questions = (
        pq.read_table(source["local_path"], columns=["question"])
        .column("question")
        .to_pylist()
    )
    if len(questions) != 1319 or len({normalize(q) for q in questions}) != 1319:
        raise ValueError("Source count or normalized duplicate policy changed")
    tokenizer_path = Path(tokenizer_path)
    inputs = {
        "pr38_panel": identity(pr38_panel),
        "source_manifest": identity(source_manifest),
        "test_parquet": bound,
    }
    for name, expected in TOKENIZER_HASHES.items():
        item = identity(tokenizer_path / name)
        if item["sha256"] != expected:
            raise ValueError("Changed native tokenizer")
        inputs["tokenizer/" + name] = item
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    # Reconstruct previous prompts before creating this independent panel.
    for retained in old["requests"]:
        ids, text = render_question(questions[retained["source_row"]], tokenizer)
        if (
            ids != retained["prompt_ids"]
            or hashlib.sha256(text.encode()).hexdigest()
            != retained["rendered_prompt_sha256"]
        ):
            raise ValueError("PR38 exact prompt reconstruction failed")
    history_path = (
        ROOT
        / "dflash_first_error_fresh_confirmation_20261027/inventory_audit/INVENTORY_AUDIT.json"
    )
    extra_path = (
        ROOT
        / "dflash_survival_confirmation_20261111/preparation/input_evidence/ADDITIONAL_EXPOSURE.json"
    )
    history = read(history_path)
    exposure = defaultdict(list)
    for item in history["primary_completed_execution_sets"]:
        for i in item["source_rows"]:
            exposure[i].append(item["directory"])
    for key in (
        "oct22_additional_completed_source_rows",
        "async_final_additional_completed_source_rows",
    ):
        for i in history[key]:
            exposure[i].append(str(history_path) + "#" + key)
    for item in read(extra_path)["sets"]:
        for i in item["source_rows"]:
            exposure[i].append(dict(name=item["name"], evidence=item["evidence"]))
    inputs.update(
        history_inventory=identity(history_path),
        additional_exposure=identity(extra_path),
    )
    requests = []
    old_hashes = {r["prompt_ids_sha256"] for r in old["requests"]}
    for ordinal, index in enumerate(selected):
        ids, text = render_question(questions[index], tokenizer)
        if not ids or len(ids) + 2048 + 16 > 4096 or digest(ids) in old_hashes:
            raise ValueError(
                "Selected prompt violates context/held-out contract; no replacement"
            )
        requests.append(
            dict(
                request_id=f"successor-gsm8k-{index:04d}",
                dataset="gsm8k",
                source_row=index,
                source_id=f"gsm8k:main:test:{index}",
                dataset_id=f"gsm8k:main:test:{index}",
                source_split="test",
                source_revision=REVISION,
                original_index=index,
                request_index=ordinal,
                panel_ordinal=ordinal,
                prompt_ids=ids,
                prompt_ids_sha256=digest(ids),
                rendered_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                normalized_question_sha256=hashlib.sha256(
                    normalize(questions[index]).encode()
                ).hexdigest(),
                max_new_tokens=2048,
                stop_token_ids=STOP_IDS,
                collect_feedback=False,
                selection_sha256=row_key(index)[0],
                prior_project_execution="documented" if exposure[index] else "unknown",
                prior_project_execution_evidence=exposure[index],
            )
        )
    matched = matched_subset(requests)
    audit.update(
        inputs=inputs,
        pr38_prompt_reconstructions=768,
        inherited_exclusions=old["selection"]["exclusions"],
        inherited_reservation_disposition=old["selection"]["reservation_disposition"],
        question_normalization="NFKC + casefold + collapsed whitespace; previously verified equivalent eligibility",
        source_answers_used=False,
        read_columns=["question"],
    )
    panel = dict(
        schema="successor_shuffle_panel_v1",
        count=len(requests),
        requests=requests,
        selection=audit,
        input_only_selection=True,
        source_revision=REVISION,
        held_out_scope="Training fork, full development and SURVIVAL panels; not project-unused",
        matched_start_request_ids=[r["request_id"] for r in matched],
        selection_precedes_reference_lookup=True,
        prior_project_execution_counts=dict(
            documented=sum(bool(exposure[i]) for i in selected),
            unknown=sum(not exposure[i] for i in selected),
        ),
        model_work=0,
    )
    panel["binding"] = digest(panel)
    verify_panel(panel)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [
        ("PANEL.json", panel),
        ("EXCLUSION_AUDIT.json", audit),
        (
            "MATCHED_START_PANEL.json",
            dict(
                requests=matched,
                request_ids=[r["request_id"] for r in matched],
                count=64,
                parent_one_block_budget=64,
                panel_binding=panel["binding"],
                selection_prefix=MATCHED_PREFIX,
                child_extra_generations=0,
            ),
        ),
        (
            "EXPOSURE.json",
            dict(
                counts=panel["prior_project_execution_counts"],
                requests=[
                    {
                        k: r[k]
                        for k in (
                            "request_id",
                            "source_row",
                            "prior_project_execution",
                            "prior_project_execution_evidence",
                        )
                    }
                    for r in requests
                ],
                selection_use=False,
            ),
        ),
    ]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return panel


def prepare_references(panel, output):
    """Run exact retained PR25/VFD validation only after the new panel is sealed."""
    from experiments.block_depth.references import prepare_references as reuse

    verify_panel(panel)
    refs, audit = reuse(
        panel["requests"],
        pr25_root=ROOT / "dflash_kplus_20261103/retrieved_001/production_001",
        pr25_inventory=ROOT
        / "dflash_kplus_20261103/retrieved_001/REMOTE_INVENTORY.json",
        vfd_root=ROOT
        / "dflash_vfd_second_pass_20261104/retrieved_001/evidence/production_001",
        vfd_inventory=ROOT
        / "dflash_vfd_second_pass_20261104/retrieved_001/PRODUCTION_INVENTORY.json",
    )
    audit.update(
        panel_binding=panel["binding"], selection_sealed_before_reference_lookup=True
    )
    out = Path(output)
    for name, value in [
        ("REUSED_CANONICAL.json", refs),
        ("CANONICAL_REUSE_AUDIT.json", audit),
    ]:
        with (out / name).open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
    return refs, audit
