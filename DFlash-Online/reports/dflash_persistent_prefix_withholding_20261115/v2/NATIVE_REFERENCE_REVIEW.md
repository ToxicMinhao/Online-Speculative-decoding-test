# Installed native-verifier reference route review

Status: source/provenance review complete; GPU parity and resource qualification pending. No new model calls. The reused reference audit separately binds104 selected requests and two warmups to318 retained frozen native executions. Twenty-four selected requests require new fixed-K6 reference generations. These references are `native_greedy_target_verifier`, not independent target-only AR.

The installed A800 sources were retrieved read-only from the pinned `adaflash_author` environment. Current `speculative/dflash_worker.py` and `dflash_info.py` hashes exactly equal the historical full-K runtime pins. The CPU dependency inspection checks all seven inherited runtime pins. Additional first-token/graph source files are pinned in this version's source receipt.

First-token chain, reviewed directly:

- `online_feedback/sglang_worker.py:NativeWorker.generate` sends temperature0,top_p1,top_k1,cap2048,ignore_eosFalse and the native stop IDs.
- Installed `speculative/dflash_worker.py:1504` delegates prefill to the frozen target worker and returns its `next_token_ids` unchanged.
- `managers/tp_worker.py:500` obtains those IDs from `model_runner.sample` for ordinary non-prefill-only requests.
- `model_executor/model_runner.py:2612–2648` applies the ordinary sampling metadata and calls the sampler; no penalties, grammar or logits bias are requested by this protocol.
- `sampling/sampling_batch_info.py:171` sets all-greedy for top_k<=1. `layers/sampler.py:105–107` chooses `torch.argmax(logits,-1)`.
- The hook binds this prefill output to the first response token. Its verification hook independently records actual target-logit argmax values. `RequestRecorder.verification` checks canonical prefix, first rejection replacement, bonus commit, clipping and output coverage.
- `experiments/o_half_matched_lr/archive.py:full_trace_errors` separately reconstructs each accepted-prefix/replacement commit from retained proposals and target observations. `online_feedback/native.py:validate_generation` checks returned tokens and EOS/cap semantics.

Native source consistency and prior qualification are stronger than candidate agreement, but do not constitute a fresh independent target-only AR reproduction. New references use only authenticated original K6 and never a candidate event8 endpoint. Existing latency/acceptance outcomes do not enter selection or the new comparison.

Finite initialization policy: retain the existing maximum batch8, C1, block16 graph settings. The installed `CudaGraphRunner.capture_one_batch_size` performs exactly two warmup model forwards and one captured model forward. Target captures token counts1..16; draft captures16. Batch sizes are bounded by8 and the request pool; capture-hidden modes are bounded byNULL/LAST/FULL. The conservative lifetime bound is408 graph-shape calls (=8*(16+1)*3),1224 existing model forwards per engine initialization. The observer records actual calls/shapes and stops on a changed routine; it introduces no model calls and no complete-response warmups. A800 capability8.0 prevents the installed capability>=9 FlashInfer autotune branch. cuBLAS initialization is also separately identified in the source (fixed small matrix multiplication). Normal graph replays and draft/target decisions remain in native traces. Engine initialization, recapture, loading and residency are charged to shared wall/card budgets.

The existing two training backward warmups per initialization remain separately counted with zero Adam steps and exact state/RNG restoration. No scientific tolerance, precision, loss, clipping or serving configuration is changed.
