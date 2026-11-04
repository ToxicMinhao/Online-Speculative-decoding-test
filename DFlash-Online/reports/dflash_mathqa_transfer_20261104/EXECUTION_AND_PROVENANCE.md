# Execution and provenance

Execution commit: `12583640baf2210f767e108e3632c0e5f9b8caaa`; branch `codex/mathqa-serving-transfer`; PR #26. Scientific run: `/data/sources/minhao/runs/dflash_mathqa_transfer_20261104/production_002`.

## Fixed inputs and serving

The official MathQA archive is pinned to repository revision `4f958013c164a40745a5b771dfb005df308deb4a`, Git blob `193fb150706468110459ddcdc52ad48e3f0b7491`, SHA-256 `7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a`. Its 2,985-row test split yielded 2,984 eligible requests; source row 0 was excluded for an earlier ShareGPT input overlap. Selection checked 66,332 distinct local inventories and 1,622 duplicate copies, plus 7,935 remote inventory files. This is available-inventory exclusion, not a claim of exhaustive unknown-history exclusion.

The first 1,024 eligible rows in ascending SHA-256 order of `mathqa-transfer-v1:` plus decimal source row were sealed before any generation. User text contains only stripped problem and options; no answers, rationale, formulas or categories. The retained PR #25 wrapper disables reasoning. Prompt lengths are 49–221 tokens. IDs, lengths and hashes are public in REQUEST_IDENTITIES.csv; exact prompt/token arrays and exclusion evidence remain private.

Target: Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`. Released draft family: z-lab/Qwen3-8B-DFlash-b16 revision `9b41424b7109f9c5413454f481b09a82b85333f4`. The retained event-58 exports were resolved from pinned PR #23/#25 paired tables and authenticated against historical receipts:

| Export | model.safetensors SHA-256 |
|---|---|
| O58 | `796ec9d486046d43a2b91bd0ddf32b33eb3183b7d924e9dd9ab6bc4c6b0d0856` |
| K58 | `47a3112244fdbdc17e01c470c55bf1913b302b194fbf99d1e13239c642df859b` |
| K+58 | `4bd8e872e4b3976b0da257f0b181f0f27c8b369e2faf386d7d833de9a1c87d67` |

Each weight file contains 2,097,259,104 bytes. Pre/postflight and load receipts confirm unchanged identities. Serving used one A800 GPU, pinned native SGLang/FA3, BF16 greedy decoding, block 16, 2,048 output tokens or EOS, context 4,096, concurrency 1, capture disabled, and full idle cache flush before every request/publication. The worker used Python 3.12.14, Torch 2.9.1+cu130, Transformers 4.57.1 and CUDA 13.0. The historical RUNTIME_PIN is an environment provenance record; current source/storage are established by fresh SOURCE/PREFLIGHT/POSTFLIGHT receipts, not old fields within that pin.

All six arm orders occur eight times under the fixed 16-block rotation. There was no competing training or transfer during timing. Every new request uses its O58 repeat-zero output as canonical; this validates cross-arm agreement under the native verifier, not against a separate target-only decode.

## Startup and access history

The first CPU-only startup (`production_001`) stopped before model work because an inventory exclusion checked the absolute A800 parent `/data/sources/` instead of workspace-relative path components. The focused fix is the execution commit above; 40 CPU tests pass, and the preselected panel hash remained `10a557a006e77dfbdee0e7284f27ad808ace50c11e6040ea431d2a3c5f5a966f`. The original failure and source bytes remain preserved. This was not a scientific retry or a changed request panel.

During preparation, six stale reporting/test/documentation files in the working mirror were synchronized to their already-published PR #25 versions, retaining the original bytes. No algorithm, checkpoint or decoding setting changed. A subsequent SSH outage prevented monitoring briefly; the original detached production_002 process continued. It was not restarted. Successful wrapper, campaign, report and postflight exit receipts are retained.

## Storage and retention

Before launch, six exactly redundant A800 tensor copies were removed only after fresh byte-hash verification of retained Fenrir copies, freeing 20,975,529,984 bytes. Exact paths and retained-copy mappings are in the private CLEANUP_EXEC receipt. No unique evidence or global replay capability was lost. Free space reached 68.869 GB and passed the 65.216 GB full-run budget including the 8 GiB reserve.

After serving completed, the entire run and failed startup/deployment evidence were copied privately to Fenrir and all 27,767 paths and SHA-256 values verified. No A800 source was deleted during retrieval. The final retrieval receipt records 19.946 GB free on Fenrir; the final A800 check recorded approximately 66.786 GB. Raw traces and token records remain on both hosts. Public tables exclude prompt/answer text and token arrays.

## Audit limits

The statistical and raw-record reviews are separate same-team calculations with author overlap disclosed. No GPU replication or new target-only reference was performed. Native environment/load/cache checks are retained runtime receipts, not a new decoding of all model tensors by the reviewers. The confidence intervals describe request variation for this fixed panel and these seed-17 fitted checkpoints, not training-seed uncertainty or a general guarantee of transfer.
