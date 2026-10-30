# Neutral qualification repair: source diagnosis

ARS experiment-agent / bounded implementation debugging, October 29, 2026. Read-only audit by the numerical-core author. No model calls, optimizer applications, GPU tests or remote operations were performed. The user authorizes repairing the neutral mismatch and rerunning bounded qualification only; the algorithm, precision, tolerances and stopped production campaign remain unchanged.

Reviewed checkout: `a55602c869d2dced6789368a343e27e0ac7a7dbe`; failed execution: `9fd9fe6f7db9796a49e2a6879b5a79e7f01fbd22`. The inspected SpecForge Flex wrapper, DFlash attention and block-construction files match `raw/sealed/SEAL.json` in the retained private run exactly. PyTorch implementation details below come from the local installation reporting the same recorded training version, `2.13.0+cu130` (`git_version=cf30153c4c131c8164ee7798e5022d810682e2cb`); byte identity of those installed PyTorch source files with the remote installation was not checked in this read-only local task.

## 1. Confirmed harness asymmetry: shared compiler history

This is the strongest actionable source finding, **not yet a confirmed cause of the tensor mismatch**.

- `specforge/modeling/draft/flex_attention.py` stores both compiled FlexAttention and compiled block-mask construction in module-level singleton instances. FlexAttention is created through `torch.compile(flex_attention)` without an explicit `dynamic` setting. The same singleton is reused across draft-model instances.
- `experiments/kw_v1/qualify.py` runs all four K responses, closes K, and constructs KW in the same process. `SpecForgeEventTrainer.close()` releases the trainer/distributed runtime and empties the CUDA allocator cache; it does not reset Dynamo or either Flex singleton.
- The model/Adam/RNG restore checks therefore do not restore the compilation history that existed before K's first response. KW's first response follows four K shapes through the same compiled wrapper.

The query sizes are derived from the retained anchor-array lengths and the unchanged block size 16. The KV sizes follow DFlash's concatenation of context and noisy draft positions. These are source-derived dimensions, not recorded generated-kernel identities.

| K order | Response | Context tokens | Anchors | Query length | KV length |
|---|---|---:|---:|---:|---:|
| 1 | train_00947 | 647 | 512 | 8,192 | 8,839 |
| 2 | train_00916 | 532 | 455 | 7,280 | 7,812 |
| 3 | train_00922 | 514 | 426 | 6,816 | 7,330 |
| 4 | train_00997 | 514 | 365 | 5,840 | 6,354 |

Sources: `raw/sealed/QUALIFICATION_INPUTS.json`, `raw/implementation_qualification/K_RESPONSES.json`, `OnlineDFlashModel._forward_draft_blocks`, and `Qwen3DFlashAttention._compute_qkv`.

The matching-version local `torch/_dynamo/config.py` defaults to `assume_static_by_default=True` and automatic dynamic shapes. Its source explicitly describes compiling the first frame statically and recompiling dimensions that change as dynamic. Thus this shape sequence can give K and KW different compilation specializations/history. Neither the retained launch script nor the local mirrored environment setup sets an automatic-dynamic-shape override. The retained qualification log contains no recompile/guard/kernel diagnostics, so it does **not** prove which specialization served any response or that different kernels caused the discrepancy.

A fresh process for each compared arm removes this known process-history asymmetry without changing K/KW numerical definitions or spending additional model warmups. It is a justified qualification-harness repair. A passing repeat would qualify the repaired comparison; identifying a particular static/dynamic kernel as the root cause still requires the corresponding diagnostic evidence.

## 2. Confirmed evidence gap: first differing stage was not recorded

The original run compared only the scientific endpoint. It did not persist K's native pre-clip gradient or a full-model before/after-preview isolation comparison. Its mismatch list also stops after 100 paths. Therefore equality of the reported losses, anchors and scalar gradient norm cannot locate the first differing tensor operation.

The proposed repaired qualification should retain compact, complete mismatch counts and magnitudes for the native pre-clip K/KW gradients, then check live parameter/Adam/gradient identity across previews. These observations can use the already authorized ordinary backwards and scratch proposals; they require no extra model calls. They distinguish an incoming-gradient discrepancy from a later preview/live-update discrepancy.

## 3. Hook, accumulation and optimizer audit

No actionable direct alias or mutation defect was found in the numerical core. The hook clones each incoming contribution and returns `None`; the actual neutral weighting branch skips `grad.copy_`; scratch parameters, gradients and state tensors are cloned. The native live optimizer remains the original adapter. These findings are unchanged from [the retained first failure analysis](../results_20261029/SOURCE_FAILURE_ANALYSIS.md).

The installed `AccumulateGrad` source uses the supplied gradient to initialize/clone an absent `.grad`, then performs native in-place addition when `.grad` already exists under ordinary backward. KW does not replace this accumulation with its captured contributions. Only the attention subroutine is compiled in the inspected author path; the Python leaf capture hooks are outside that compiled function. This audit does not prove that adding hooks has no runtime effect, but it identifies no hook arithmetic that explicitly rewrites the neutral accumulated gradient.

An inherited detail deserves separation from this repair: SpecForge places `no_sync()` around backward only, while PyTorch's DDP documentation requires forward inside that context to suppress synchronization. Both arms inherit this behavior, and the qualification has world size one. It is not a KW-specific source change or a demonstrated explanation for this failure, and should not be changed as part of the neutral repair.

The DFlash source explicitly selects general Triton FlexAttention for the pinned Torch version. The inspected Triton backward template writes each dQ tile with `tl.store` and accumulates grouped-query dK/dV in an explicit loop. This does not establish whole-backward determinism, but does not support casually attributing the failure to an unspecified atomic dQ kernel either.

## Current conclusion

A process-history asymmetry is confirmed and can be removed while preserving the fixed algorithm. Its causal role remains unmeasured. Exact native-gradient and preview-isolation observations are necessary to interpret the bounded rerun. No production work, tolerance relaxation, precision substitution, or unbounded diagnostic sequence is justified by this source review.
