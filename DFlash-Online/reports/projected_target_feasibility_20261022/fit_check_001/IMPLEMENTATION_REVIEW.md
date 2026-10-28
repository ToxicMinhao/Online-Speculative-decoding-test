# Bounded fit comparison: implementation and review

Uses the ARS experiment-agent workflow and an independent source review. The maintained entry is A800 `/data/sources/minhao/DFlash-Online/experiments/projected_target/fit_check.py`; `fit_metrics.py` contains pure diagnostic arithmetic. Fenrir DFlash-Research is unchanged.

The driver builds the original SpecForge model, loads only the frozen target embedding/head, and loads the retained C/P exports into strict copies of the original draft architecture. It uses `_ReferenceBlocks` and `_forward_draft_blocks` from the existing implementation. Exact saved anchors replace sampling. Labels, masks, position weights, BF16 heads and eight-block vocabulary chunks match the original objective. No optimizer or generation is constructed.

Two independent reviews checked model loading, conditioning, all 32 exposure receipts, fixed reference membership, source export identities and response normalization. Eighteen focused CPU tests cover KL arithmetic, numerical zeros/ties, masking, weights, repairs/damage, EOS, blocking errors and cap censoring. The first actual F layout must reproduce the original initial P softCE within predeclared `atol=2e-5, rtol=2e-5`; this checks execution fidelity, not adaptation performance.

Storage admission: A800 available 13,474,361,344 bytes; reserve 8 GiB and diagnostic output budget 256 MiB. Both GPUs were idle. No deletion or transfer of large artifacts was needed. Retained four exports and sixteen feature files were present.

Before model execution, an outdated remote fixture failed after the cap-boundary convention was clarified; the corrected fixture uses a strictly internal block. A subsequent shell wrapper failed because `/usr/bin/time` was unavailable. Both logs are retained. Neither attempt constructed a model or evaluated a layout. The final wrapper uses the driver's built-in monotonic timers, CUDA peak counters and Python resource accounting. These repairs do not change scientific settings.

## Completed independent result audit

All 96 forwards and 6,144 block scores completed. Both independent reviewers recovered the primary KL and prefix results. One independently reconstructed every block prefix from position rows and saved canonical F records; all passed. Fixed labels, weights, reference partitions and target statistics agree across all 29,928 aligned exposures. These represent 4,197 distinct sampled source-token positions, with repeated/overlapping conditioning. The first-source pre-update softCE matches the historical value exactly. Final P KL exceeds F on every layout, including each fixed correct/wrong partition. No implementation arithmetic issue was found; no causal optimizer diagnosis follows from final-checkpoint data alone.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
