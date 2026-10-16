"""Lossless, pinned Llama-3.1 conversation framing (no model imports).

The published chat template strips user whitespace. These selected inputs must
instead retain it exactly. Only that trim is bypassed: headers, system text and
generated assistant token IDs retain the qualified serving contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
TOKENIZER_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
TOKENIZER_SHA256 = {
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
}
CHAT_TEMPLATE_SHA256 = "ca5b86f15c19b53c24ff18228849e06ffb677b952d85171c66d9b720d924f9bb"
INPUT_LIMIT, CONTEXT_LIMIT, OUTPUT_CAP, BLOCK_SIZE = 31744, 32768, 1024, 10
VOCAB_SIZE = 128256
BOS = "<|begin_of_text|>"
EOT = "<|eot_id|>"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
COMMON_SYSTEM_TEXT = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def check_input(ids):
    require(isinstance(ids, list) and 0 < len(ids) <= INPUT_LIMIT and
            all(type(x) is int and 0 <= x < VOCAB_SIZE for x in ids),
            "Complete input must contain valid token IDs and fit the input limit; truncation is prohibited")
    require(len(ids) + OUTPUT_CAP + BLOCK_SIZE - 1 <= CONTEXT_LIMIT,
            "Complete input, 1024 output tokens and final proposal allowance exceed context")


def validate_tokenizer(tokenizer):
    template = json.dumps(tokenizer.chat_template, ensure_ascii=False, separators=(",", ":"))
    require(_digest(template.encode()) == CHAT_TEMPLATE_SHA256, "Pinned chat template changed")
    require(tokenizer.bos_token == BOS, "Pinned BOS token changed")
    require(tokenizer.encode(EOT, add_special_tokens=False) == [128009], "Pinned EOT changed")
    fixture = "Tokenizer framing check."
    manual = tokenizer.encode(BOS + SYSTEM_HEADER + COMMON_SYSTEM_TEXT + EOT +
                              USER_HEADER + fixture + EOT + ASSISTANT_HEADER,
                              add_special_tokens=False)
    native = tokenizer.apply_chat_template([{"role": "user", "content": fixture}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    require(list(native) == manual, "Pinned native system/header framing differs")


def load_pinned_tokenizer(path):
    """Validate every pinned tokenizer file before CPU-only tokenizer loading."""
    path = Path(path)
    for name, expected in TOKENIZER_SHA256.items():
        require(_digest((path / name).read_bytes()) == expected, "Pinned tokenizer file changed: " + name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    validate_tokenizer(tokenizer)
    return tokenizer


def user_segment_ids(tokenizer, content):
    require(isinstance(content, str) and content, "Nonempty exact user text required")
    text = USER_HEADER + content + EOT + ASSISTANT_HEADER
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    require(tokenizer.decode(ids, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False) == text,
            "Tokenizer does not preserve exact user text, including whitespace")
    return ids


def encode_history(tokenizer, messages):
    """Encode source users and canonical generated IDs without text normalization.

    ``messages`` alternates user/content and assistant/output_ids, ending in a
    user. No source system/persona or original assistant answer is admitted.
    Assistant ``content`` is deliberately ignored: generated IDs are authoritative.
    """
    validate_tokenizer(tokenizer)
    require(isinstance(messages, list) and messages and len(messages) % 2 == 1,
            "Conversation must alternate user/assistant and end in a user")
    require(messages[0].get("role") == "user", "Source system/persona messages are prohibited")
    ids = list(tokenizer.encode(BOS + SYSTEM_HEADER + COMMON_SYSTEM_TEXT + EOT,
                                add_special_tokens=False))
    for index, message in enumerate(messages):
        if index % 2 == 0:
            require(message.get("role") == "user", "User role required at request boundary")
            ids.extend(user_segment_ids(tokenizer, message.get("content")))
        else:
            require(message.get("role") == "assistant", "Assistant role required after user")
            output = message.get("output_ids")
            require(isinstance(output, list) and 0 < len(output) <= OUTPUT_CAP and
                    all(type(x) is int and 0 <= x < VOCAB_SIZE for x in output),
                    "Original generated assistant output_ids required; retokenization is prohibited")
            ids.extend(output)
            if output[-1] != 128009:
                ids.append(128009)
    check_input(ids)
    return ids


def require_identical_input(canonical_ids, replay_ids):
    """Exact equality, not equality of decoded/normalized text."""
    check_input(canonical_ids)
    check_input(replay_ids)
    require(canonical_ids == replay_ids, "Replay input token IDs differ from canonical request")


def conservative_budget(tokenizer, prompts):
    """Bound unknown generated histories without using original source answers."""
    validate_tokenizer(tokenizer)
    require(isinstance(prompts, list) and prompts, "Prompts are required")
    prefix_n = len(tokenizer.encode(BOS + SYSTEM_HEADER + COMMON_SYSTEM_TEXT + EOT,
                                   add_special_tokens=False))
    rows = []
    for turn, prompt in enumerate(prompts, 1):
        prefix_n += len(user_segment_ids(tokenizer, prompt))
        input_n = prefix_n + (turn - 1) * (OUTPUT_CAP + 1)
        total_n = input_n + OUTPUT_CAP + BLOCK_SIZE - 1
        require(input_n <= INPUT_LIMIT and total_n <= CONTEXT_LIMIT,
                "Prepared history exceeds context budget; no silent truncation")
        rows.append({"turn": turn, "input_tokens_upper_bound": input_n,
                     "including_current_answer_and_speculation_upper_bound": total_n})
    return rows
