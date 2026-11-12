# Repair history at matched pre-update confidence

This isolated diagnostic uses frozen native K+4/K+5/K+6. It asks whether a token
newly accepted by K+5 has additional next-update rejection risk at similar K+5
confidence. It performs no training, optimizer steps or latency benchmark.

The input preparation takes the first 192 eligible requests from the original
PR39 TRUE/KPLUS5 panel, in its original order. It checks source IDs, normalized
questions and rendered inputs against events1–6 training, PR42 discovery data
and the PR34/40 GSM8K discovery panel. Raw prompts/tokens remain outside Git.

Execution has an explicit outcome boundary:

1. Authenticate a clean source checkout, pinned checkpoints/runtime and input
   seal; admit disk, GPU and host-memory capacity.
2. Observe K+5's original native vocabulary-head multiplication, preserving its
   BF16 arithmetic, shape and greedy choice. Compare each instrumented walk with
   its authenticated uninstrumented trace, and every neutral replay with the
   original block and exact reconstructed caches.
3. Replay K+4 at all K+5 starts. Keep native margins from K+5's original scores.
   Independently verify 32 hash-selected full-vocabulary rows, including a CPU
   fixture for exact score ties.
4. Seal repaired/established labels, all structural comparator pools, margins
   and nearest-margin matches before publishing K+6 into the live model.
5. Replay K+6 at the unique required starts and report exact-slot rejection,
   earlier rejection and acceptance-through-slot using the full matched
   denominator. Bootstrap requests, preserving reused controls and pair weights.

Completed requests share bitwise-checked immutable cache-prefix storage within
one live native worker. This retains the same native parent states across the
outcome boundary without generating those histories again. A storage alias is
accepted only after every original target/draft cache prefix matches exactly.
A three aggregate occupied A800-hour limit covers initialization and partial
work. Failures are preserved; there is no automatic retry or threshold tuning.

Modules:
- `prepare.py`: panel and exclusion authentication.
- `prepare_run.py`: clean-code/configuration and actual-path execution seal.
- `native_worker.py`: original-score observation, exact state retention and QA.
- `analysis.py`: outcome-blind matching and fixed request-cluster analysis.
- `run.py`: two-phase execution, resource admission and work accounting.
- `report.py`: token-free tables and complete/partial findings.

The 0.25 margin caliper, minimum100 repaired starts/40requests, absoluteSMD≤0.10,
10,000 PCG64 bootstrap draws(seed20261115), and +5percentage-point spending
threshold are fixed. No result authorizes an intervention or another run.

Local CPU checks:

```bash
PYTHONPATH=.:training:inference:tests python -m unittest \
  test_repair_history_margin_prepare test_repair_history_margin_analysis \
  test_repair_history_margin_native test_repair_history_margin_run
```

These tests do not establish GPU parity; the native run must pass its own exact
output, proposal, cache and margin checks before statistical interpretation.
