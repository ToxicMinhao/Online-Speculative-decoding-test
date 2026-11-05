# Readout, exposure and retention review

Verification status: **ANALYZED** from retained evidence, without a GPU rerun. This is a same-team review by the readout implementation author, not independent replication. Execution `0f704167b268672d980fabe704898abf9b134c39`.

All three groups completed 256 presentations and 128 AdamW steps each: 768 / 384 total, zero extra startup optimizer steps. Every step had finite nonzero gradients and changed BF16 sidecar coordinates. All decoded boundary/export/Adam/RNG checks and both resumptions passed; frozen-backbone versions remained unchanged.

Actual exposure was **49,152 anchors, 714,583 available valid proposal occurrences and 169,351 selected direct-KL occurrences**. Of 48,947 first errors, 21,763 received priority 2 and 27,184 received priority 1; 205 blocks had no error. The detached matched-successor cue selects a first-error weight; it does not add a supervised position. Priority 1 combines unavailable and mismatching successors. These are occurrence counts, not unique tokens. No triangular protection weight was introduced; original K+ positional decay and its first-error redistribution remain in use.

Readout fitting took **252.139 s**; three worker lifetimes totaled **334.486 s**, including **70.411 s** initialization. Group launches totaled **345.392 s**. Peak readout allocation was **6,204,037,120 bytes** and reservation **7,113,539,584 bytes**. READOUT collection/reconstruction/preparation was **287.639 s**, and GATE collection/reconstruction/fit **219.436 s**. These timers overlap by containment and must not be added indiscriminately.

The pass-mean loss traces decrease (group 1: 1.81577→1.41080; group 2: 1.35264→1.22607; group 3: 1.30016→1.20102), but anchors and current prediction masks change. They are not a fixed-anchor source-fit test or proof of native usefulness. Native serving uses fused residual RMSNorm while SpecForge training uses residual-complete h3 and ordinary RMSNorm. The mathematical input is aligned; bitwise equivalence is not asserted, and this distinction is not an established cause of shallow weakness. Native THREE and adaptive shallow inference share the same implementation.

The **387 completed retirement receipts cover 1,152 unique files and 12,606,855,808 bytes**: 384 consumed teacher-feature files (6,302,229,440 bytes), 384 native captures (6,284,369,600 bytes), and 384 terminal features (20,256,768 bytes). Exact paths, reasons and receipts are in `RETIRED_PATHS.csv`. They are retired, not archived; tensor replay requires reconstruction. Compact feedback/tokens/anchors, sidecar exports, optimizer/RNG recovery states, source and validity evidence remain.

The main report’s readout counts, cost nesting, source-fit limitation and retention wording agree with the evidence. No experiment code was changed during this review. `AUDIT.json` contains recomputed totals and checks; `EXPOSURE.csv` has one row per presentation, avoiding accumulation-row double counting.
