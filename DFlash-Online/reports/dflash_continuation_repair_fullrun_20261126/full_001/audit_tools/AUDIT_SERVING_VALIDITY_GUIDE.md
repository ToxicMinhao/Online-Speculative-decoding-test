# Independent native-serving validity audit

This is a prepared CPU audit tool, not a result for the ongoing BASE/TAIL full run. It imports no model, production validator or statistical reducer and starts no server work.

Run after authenticated retrieval, preserving the original sealed plan and raw files:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/minhao2/.venvs/dflash-specforge/bin/python audit_serving_validity.py \
  --plan /absolute/path/to/original/PLAN.json \
  --raw-root /absolute/path/to/retrieved/raw \
  --external-map /absolute/path/to/EXTERNAL_EVIDENCE_MAP.json \
  --output /absolute/path/to/final_audit/serving_validity_001.json
```

The external map is a JSON object mapping each original absolute evidence locator outside the run root to its authenticated local copy. It must include the source manifest, original inputs, new evaluation inputs and retained warmup canonical file as needed by the sealed plan. Original absolute paths inside the run root are relocated automatically. There is no implicit fallback to live files. Retrieve the separate native trace JSON files referenced by each raw record, along with raw records, protocol, status, canonical, activation, boundary, weight and export-identity receipts.

A successful result requires all 1,024 fresh native-F canonical records, 12,288 scored records, 96 planned warmups and any explicitly counted resume warmups. The audit independently reconstructs each block's consecutive matching prefix, committed tokens, EOS clipping, call counts, accepted-token counts and output progress. It checks exact final tokens/termination against the corresponding native-F canonical; distinct native accepted and stop-clipped accepted counters; input/rendered-token bindings; balanced activation order; repeat consistency; raw metric agreement; quiet timing boundaries; endpoint export identities; and complete/invalid/unknown accounting. Missing evidence returns `incomplete`; contradictions return `failed`. Only complete success returns lowercase `passed` with the sealed `plan_binding` and `execution_commit`.

The output path is created exclusively so an earlier failed audit cannot be overwritten. The companion `*_PRIVATE_INPUTS.json` contains original paths and fresh evidence identities; keep that input inventory private. Before publishing a failure result, inspect its diagnostic string for private file locators. A successful summary contains no prompts, response tokens, request identities or anchor arrays.

Run the separate `audit_statistics_independent.py` with the same `--plan`, `--raw-root` and `--external-map` arguments and a distinct `--output` for per-request aggregation, shared paired bootstrap and the prespecified practical/statistical decision. It independently authenticates the sealed plan, source manifest and panel input identities, then hashes the actual measurements and final analysis consumed. Its passing receipt also carries the plan binding and execution commit. It does not import the raw-validity auditor. A publication gate must require both audit receipts; the arithmetic audit alone does not validate raw outputs. Resource, optimizer/work, live endpoint tensor bytes and retained-source authentication remain companion-audit responsibilities. If an uppercase `PASSED` work/resource receipt needs a lowercase publication status, create an explicit wrapper retaining the original receipt identity and plan/source binding; do not rewrite the original audit.

Limitations are explicit: this tests agreement with native frozen DFlash, not a new target-only autoregressive reference. Rendered text/token hashes and pinned bytes are checked without rerunning tokenization. Endpoint tensor bytes are covered by live postflight, not reconstructed from compact receipts. The auditor was independently written without importing production validators, but shares project context and authorship of the serving implementation.

Readiness validation: 11 synthetic CPU tests cover a complete 1,024-canonical/12,384-serving fixture, truncation/counter distinctions, malformed or missing evidence, input/source/schedule mutations and file relocation. A separate schema check accepted 100 retained PR65 native records using each record's own output as a reference; this checks schema compatibility only and is not independent F agreement or evidence about the new full run. No GPU work was used for these checks.
