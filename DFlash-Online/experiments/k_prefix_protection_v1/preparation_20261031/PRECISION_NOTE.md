# Prelaunch precision note

CPU analysis of the completed coding study; no model execution.

| Original endpoint | Comparison | Call reduction | Joint-request 95% interval |
|---|---|---:|---:|
| Event 2 | K_vs_O | +0.6604% | [-0.0263, +1.3179]% |
| Event 2 | K_vs_F | +0.2393% | [-0.7617, +1.1293]% |
| Event 4 | K_vs_O | +0.7400% | [-0.0211, +1.5218]% |
| Event 4 | K_vs_F | +0.0342% | [-1.1362, +1.1031]% |

Historical intervals are planning proxies. Candidate-versus-K variance is unknown; these do not guarantee power. The fixed 0.5% target is a research-priority choice, not a measured deployment break-even. No panel resizing, tuning or outcome-based selection.

Both historical endpoint intervals are reported; the new campaign evaluates only event 4. Their uncertainty should be compared with the 0.5% priority target, rather than read as a power estimate for KP versus K. Each bootstrap sample jointly retains both seeds; timing repeats and seed/request rows are not independent observations.

Source: `/home/minhao2/EAGLE_SPEC/research/dflash_ok_coding_20261030/production_002/final_report/PAIRED_REQUESTS.csv`. Exact source bytes and bootstrap configuration are in PRECISION_NOTE.json.
