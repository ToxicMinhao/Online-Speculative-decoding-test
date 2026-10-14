# STABLE/SHORT dataset tools

Construct controlled request streams from original user prompts in the train splits of [Tulu personas code](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code) and [Tulu personas math](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-math). Source assistant answers are excluded. These tools do not run a model, generate answers, train, or select by model performance.

## Fixed design

Seed: **20261014**. Sources are pinned in `select_prompts.py` to code revision `1412abe88dd2976af977260788e033013449f7b2` and math revision `46add032f5203fd96efbd54d074198981ed57d25`, including exact parquet SHA256 identities.

Select128 code and128 math questions, then assign four disjoint replicas with32 of each type. Within each replica, C1–C4 and M1–M4 contain eight questions per block.

| Stream | Ordered blocks |
|---|---|
| STABLE_CODE | C1 → C2 → C3 → C4 |
| STABLE_MATH | M1 → M2 → M3 → M4 |
| SHORT_CODE_FIRST | C1 → M2 → C3 → M4 |
| SHORT_MATH_FIRST | M1 → C2 → M3 → C4 |

Turns1–8 supply initial context;9–16 are adaptation support;17–24 are evaluation;25–32 are continuation. A question always retains the same phase. Replicas1–2 are development and3–4 are reserved confirmation. Shared blocks retain identical text, IDs and order. This is task-family continuity, not a claim that questions refer to one another.

## Rebuild an existing reviewed selection

From the repository root, using Python3.10 or newer:

```bash
python datasets/stable_short/build_streams.py --selection /path/to/selected_prompts.jsonl --output-dir /path/to/streams
python datasets/stable_short/build_streams.py --selection /path/to/selected_prompts.jsonl --output-dir /path/to/streams --check
python -m unittest discover -s datasets/stable_short -p 'test_*.py' -v
```

The builder and synthetic tests use only the standard library. Outputs are `streams.jsonl`, `validation.json` and complete32-request previews `preview_R1_STABLE_CODE.md` / `preview_R1_SHORT_CODE_FIRST.md`. Different existing outputs are not overwritten. `--check` verifies byte-identical reconstruction without writes.

The selected input uses exactly these fields: `source_dataset`, `source_revision`, `source_split`, `source_row_index`, `source_id`, `label`, `prompt`, `question_id`, `prompt_sha256`, `replica`, `block`, `position_in_block`, `sampling_rank`, `seed`. Prompts remain byte-exact UTF8 strings; metadata is never inserted into them. Placements add `role`, `split`, `stream_id`, `stream_type`, `turn`, `phase`. Unknown input fields are rejected to prevent answer contamination.

## Prepare a new selection from the pinned sources

Only candidate extraction needs `pyarrow` (the executed construction used25.0.0). The three source downloads total400,222,466bytes. Use an appropriate existing environment or install that dependency in your chosen environment.

```bash
python datasets/stable_short/select_prompts.py candidates --cache-dir /path/to/source_cache --output-dir /path/to/selection --count-per-label 192
# Inspect each candidate prompt and create the quality-review JSON files described below.
python datasets/stable_short/select_prompts.py select --candidates /path/to/selection/candidates.jsonl --reviews /path/to/review_code.json /path/to/review_math.json --output-dir /path/to/selection
```

Each source uses an independent `Random(20261014).shuffle` of all original row indices, in pinned parquet-file order. Review candidates in increasing sampling rank. The selector takes the first128 eligible per label; invalid examples are replaced by subsequent eligible candidates without resampling or rewriting text. No question difficulty, output-length preference, draft acceptance or model result is a selection criterion.

A review file is an object with a `decisions` array. Each entry contains `question_id`, exact `prompt_sha256`, `decision` (`accept` or `reject`) and a reason. Other top-level review provenance can identify the reviewer and criteria. Quality criteria: English, self-contained, source-consistent task label, no malformed or duplicate question, and no missing required files/images/prior conversation. A generic request to implement a function accepting a file is not automatically a missing-file question. Material contradictions or missing essential inputs should be documented.

Semantic eligibility requires inspection; this is not automated quality certification. A changed review can change the selected set. Reproducing an already chosen set exactly uses its saved selection, not a new review. Generated prompts, selections, review decisions, previews and result artifacts are intentionally **not included in this code-only repository**.

## Checks and limits

Validation requires256 unique questions,128 of each label,16 streams,32 turns per stream,512 placements, exact shared-block reuse, no within-stream or cross-replica duplicates, and no support/evaluation overlap. Selected prompts are checked both exactly and after whitespace normalization; selection also casefolds its duplicate key. It does not certify absence of all semantic near-duplicates. Lengths are reported as characters and whitespace-separated words, not tokenizer counts.

Tests use synthetic text and cover altered prompts/review hashes, answer contamination, missing reviews, seeded indices, duplicate questions, wrong blocks, phase leakage, metadata types, Unicode line separators and deterministic no-overwrite rebuilding. Model-performance conclusions require separate experiments.
