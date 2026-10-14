# Two-update dataset extension

This standard-library-only builder reuses the saved 256-question selection and
the validation contract in `../stable_short/build_streams.py`. It preserves all
four original streams' prompt text, IDs and order, changes phase metadata, and
adds one RANDOM_MIX stream per replica. It does not download or reselect data.

| Turns | Phase | Intended chronology |
|---|---|---|
| 1–8 | `context` | Build conversation context |
| 9–16 | `support` | First update uses these responses after turn 16 |
| 17–24 | `evaluation_then_support` | Record evaluations before these responses become second-update support |
| 25–32 | `final_evaluation` | Evaluate after the second update; no further update |

Each RANDOM_MIX stream uses a private `random.Random(20261014 + replica)`.
It shuffles a list of 16 `code` labels followed by 16 `math` labels. For each
eight-turn phase in order, it samples the required number of questions without
replacement from the matching Ck pool, then Mk pool, each sorted by
`position_in_block`. It consumes those sampled lists in shuffled label order.
The whole stream is balanced; phases need not be. For example, replica 1 has
4, 4, 2 and 6 code questions in its four phases.

The result has 20 streams, 640 placements and 256 unique questions. Each question
appears twice in the original streams and may appear once more in RANDOM_MIX;
128 questions have two placements and 128 have three. Every placement stays in
the question's original phase and replica. Original shared blocks remain exact.
Replicas 1–2 are development; replicas 3–4 remain reserved for confirmation.

Run from the repository root, pointing to the retained selection and a separate
output directory outside this public code checkout:

```bash
python datasets/two_update/build_streams.py \
  --selection /absolute/path/to/selected_prompts.jsonl \
  --output-dir /absolute/path/outside/checkout/two_update_dataset
python datasets/two_update/build_streams.py \
  --selection /absolute/path/to/selected_prompts.jsonl \
  --output-dir /absolute/path/outside/checkout/two_update_dataset --check
python -m unittest discover -s datasets/two_update -p 'test_*.py' -v
```

Existing different artifacts are never overwritten. `--check` rebuilds and
compares exact bytes without writing. Outputs are `streams.jsonl`,
`manifest.json`, `validation.json` and 20 complete stream previews. The manifest
records source revisions, selection and stream hashes, selected IDs, mixing
seeds, source-code hashes and chronology. Generated prompts, previews, manifests,
selection reviews, responses and results are not part of this code repository.

The importable API is `build_streams(selection: list) -> list` and
`validate_streams(selection, streams) -> None`; `read_selection(path)` returns
`(rows, raw_bytes)`. Each placement retains the original selection fields plus
`role`, `split`, `stream_id`, `stream_type`, `turn` and the new `phase` value.
No reference-answer field is accepted. Selection seed metadata remains 20261014;
per-replica RANDOM_MIX seeds are separate manifest metadata.

These checks establish dataset structure and deterministic construction. The
experiment runner must separately enforce evaluation-before-support, causal
generated history, response-only loss masks and no update-2 replay of turns
9–16. Dataset validation does not qualify model behavior or establish gains.
