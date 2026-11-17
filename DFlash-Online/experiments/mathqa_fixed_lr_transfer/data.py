"""Input-only MathQA identities and the fixed two-arm held-out schedule.

Historical seals stay immutable. New deduplication preserves case and punctuation
and never rewrites the model prompt; responses and performance are not inputs.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import re
import unicodedata

SELECTION_PREFIX = "kplus_mathqa_lr_transfer_20261116_v1"
SOURCE_REVISION = "4f958013c164a40745a5b771dfb005df308deb4a"
SOURCE_BLOB = "193fb150706468110459ddcdc52ad48e3f0b7491"
PANEL_SIZE, WARMUP_SIZE = 256, 2
FINGERPRINT_FIELDS = ("normalized_content_sha256", "normalized_question_sha256",
                      "normalized_prompt_sha256", "prompt_ids_sha256")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_digest(tokens):
    return digest(json.dumps(tokens, separators=(",", ":")))


def fingerprints(question, options, rendered_prompt, prompt_ids):
    """A newline separates question/options before the specified normalization."""
    return dict(normalized_content_sha256=digest(normalize(question+"\n"+options)),
                normalized_question_sha256=digest(normalize(question)),
                rendered_prompt_sha256=digest(rendered_prompt),
                normalized_prompt_sha256=digest(normalize(rendered_prompt)),
                prompt_ids_sha256=token_digest(prompt_ids))


def rank_key(source_id, normalized_content_sha256):
    return digest(SELECTION_PREFIX+"|"+source_id+"|"+normalized_content_sha256)


def rank_eligible(rows):
    """Rank an already audited eligible population by the prescribed input key."""
    return sorted(rows, key=lambda r: (rank_key(r["source_id"], r["normalized_content_sha256"]),
                                       r["source_id"]))


def seal_requests(eligible):
    """Take exactly the first256+2; this function neither relaxes nor retries."""
    ranked = rank_eligible(eligible)
    if len(ranked) < PANEL_SIZE+WARMUP_SIZE:
        raise ValueError("Fewer than258 audited eligible requests")
    roles = {}
    for role, selected in (("scored", ranked[:PANEL_SIZE]), ("warmup", ranked[PANEL_SIZE:PANEL_SIZE+WARMUP_SIZE])):
        rows = []
        for ordinal, original in enumerate(selected):
            row = deepcopy(original)
            row.update(role=role, panel_ordinal=ordinal, ordinal=ordinal, request_index=ordinal,
                       request_id=f"mathqa-lr-transfer-{original['source_row']:04d}",
                       original_index=original["source_row"], max_new_tokens=2048,
                       stop_token_ids=[151645,151643], collect_feedback=False,
                       dataset="mathqa", selection_key=rank_key(original["source_id"], original["normalized_content_sha256"]))
            rows.append(row)
        roles[role] = rows
    return roles


def validate_request(row, role, ordinal):
    if (row["role"] != role or row["panel_ordinal"] != ordinal or row["ordinal"] != ordinal
            or row["request_index"] != ordinal or row["max_new_tokens"] != 2048
            or row["stop_token_ids"] != [151645,151643] or row["collect_feedback"] is not False
            or row["dataset"] != "mathqa"):
        raise ValueError("New held-out request contract differs")
    if (type(row["source_row"]) is not int or not 0 <= row["source_row"] < 2985
            or row["source_id"] != f"mathqa:test:{row['source_row']}"
            or row["original_index"] != row["source_row"]
            or row["request_id"] != f"mathqa-lr-transfer-{row['source_row']:04d}"):
        raise ValueError("New held-out source identity differs")
    ids = row["prompt_ids"]
    if not ids or len(ids)>2048 or any(type(x) is not int or x<0 for x in ids):
        raise ValueError("Malformed or over-context original prompt IDs")
    expected = fingerprints(row["question"], row["options"], row["rendered_prompt"], ids)
    if any(row[key] != value for key,value in expected.items()):
        raise ValueError("Input fingerprint differs")
    if row["selection_key"] != rank_key(row["source_id"], row["normalized_content_sha256"]):
        raise ValueError("Wrong fixed hash rank")


def validate_manifests(panel, warmups, adaptation):
    """Check the new schema, with no old-panel or zero-reference assumptions."""
    scored, warm = panel["requests"], warmups["requests"]
    if len(scored)!=256 or len(warm)!=2:
        raise ValueError("Require the complete new256+2 panel")
    for role, rows in (("scored",scored),("warmup",warm)):
        for ordinal,row in enumerate(rows):
            validate_request(row,role,ordinal)
    combined = scored+warm
    if len({r["request_id"] for r in combined})!=258 or len({r["source_id"] for r in combined})!=258:
        raise ValueError("Duplicated scored/warmup identity")
    for key in FINGERPRINT_FIELDS:
        if len({r[key] for r in combined})!=258:
            raise ValueError("Duplicated held-out normalized input")
    if [r["source_id"] for r in combined] != [r["source_id"] for r in rank_eligible(combined)]:
        raise ValueError("Held-out hash order changed")
    train = adaptation["train_requests"]
    if (adaptation["start_ordinal"]!=512 or adaptation["stop_ordinal"]!=768
            or len(train)!=256 or [r["request_id"] for r in train]!=[f"train_{i:05d}" for i in range(512,768)]
            or len({r["source_id"] for r in train})!=256):
        raise ValueError("Original MathQA continuation stream changed")
    for i,row in enumerate(train,512):
        if (row["source_id"].split(":")[:2]!=["mathqa","train"] or row["request_index"]!=i+1
                or row["max_new_tokens"]!=2048 or row["stop_token_ids"]!=[151645,151643]
                or row["collect_feedback"] is not True or len(row["prompt_ids"])+2048>4096):
            raise ValueError("Original continuation input contract differs")
    for field in ("source_id", "prompt_ids_sha256"):
        historical={r[field] for r in train}
        if any(r[field] in historical for r in combined):
            raise ValueError("Held-out requests overlap continuation inputs")
    return dict(status="passed",scored=256,warmup=2,adaptation=256,new_canonical_generations=258,
                no_new_model_work=True)


def evaluation_schedule(requests):
    if len(requests)!=256 or [r["panel_ordinal"] for r in requests]!=list(range(256)):
        raise ValueError("Require the fixed256-request scored panel")
    result=[]
    for repeat in range(3):
        for block in range(4):
            arms=("N","S") if (repeat+block)%2==0 else ("S","N")
            for position,arm in enumerate(arms):
                result.append(dict(activation=len(result),repeat=repeat,block_ordinal=block,
                    position=position,method=arm,request_ordinals=list(range(block*64,(block+1)*64))))
    return result


def history_matches(value, rendered_rows, source_contents=None):
    """Match historical input fields only; outcome and catalog fields are ignored.

    source_contents maps pinned source IDs from any split to (question,options).
    It lets an original training ID exclude equal held-out content without using
    its answer. The caller retains each consumed inventory's path and byte hash.
    """
    ignored={"answer","response","output","completion","reference_answer","generated_text","output_text",
             "response_text","assistant","metrics","exclusions","rejections","candidate_pools","sources",
             "source_catalog","history_files","rationale","correct","annotated_formula","linear_formula",
             "category","unselected_eligible_ids","eligible_order"}
    text_fields={"text","prompt","content","value","user","user_text","question","problem","user_message",
                 "rendered_prompt","prompt_text","instruction","normalized_stem"}
    token_fields={"prompt_ids","input_ids","prompt_token_ids","rendered_prompt_token_ids"}
    lookup=defaultdict(set);by_source={r["source_id"]:r["source_row"] for r in rendered_rows}
    for row in rendered_rows:
        for text in (row["question"],row["question"]+"\n"+row["options"],row["user_message"],row["rendered_prompt"]):
            lookup[("text",normalize(text))].add(row["source_row"])
        lookup[("token",row["prompt_ids_sha256"])].add(row["source_row"])
        for key in ("normalized_content_sha256","normalized_question_sha256","rendered_prompt_sha256","normalized_prompt_sha256"):
            lookup[("hash",row[key])].add(row["source_row"])
    found=defaultdict(set)
    def match_text(text,reason):
        for variant in [text,*re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>",text,re.S)]:
            for candidate in (variant,re.split(r"\n\s*Options\s*:",variant,maxsplit=1)[0]):
                for i in lookup.get(("text",normalize(candidate)),()):found[i].add(reason)
    def scan(item,field=""):
        name=field.lower()
        if name in ignored:return
        if isinstance(item,dict):
            if item.get("role",item.get("from")) in ("assistant","gpt","chatgpt","bing","bard"):return
            for key,value in item.items():scan(value,key)
        elif isinstance(item,list):
            if name in token_fields and item and all(type(x) is int for x in item):
                for i in lookup.get(("token",token_digest(item)),()):found[i].add("exact_prompt_token_ids")
            else:
                for child in item:
                    if not isinstance(child,(int,float,bool)):scan(child,field)
        elif isinstance(item,str):
            if name in text_fields:match_text(item,"normalized_input_text")
            if name in ("prompt_ids_sha256","prompt_token_ids_sha256","rendered_token_sha256"):
                for i in lookup.get(("token",item),()):found[i].add("exact_prompt_token_hash")
            if name in ("rendered_prompt_sha256","normalized_content_sha256","normalized_question_sha256"):
                for i in lookup.get(("hash",item),()):found[i].add("input_content_hash")
            if name in ("source_id","dataset_id","request_id","id","source_key"):
                key=item.replace("mathqa_test:","mathqa:test:")
                if key in by_source:found[by_source[key]].add("source_id")
                if source_contents and key in source_contents:
                    question,options=source_contents[key]
                    for text in (question,question+"\n"+options):match_text(text,"source_content_duplicate")
    scan(value)
    return {k:sorted(v) for k,v in found.items()}
