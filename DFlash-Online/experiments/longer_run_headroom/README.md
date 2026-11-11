# Longer-run headroom diagnostic

Frozen PR34 KPLUS4/KPLUS6 diagnostic, not a training experiment or serving benchmark.

1. `python -m experiments.longer_run_headroom.census --evidence RETAINED --output CENSUS` authenticates only parent walks, reconciles the census, and seals hash-selected later starts before child outcomes are read.
2. `python -m experiments.longer_run_headroom.prepare --evidence RETAINED --census CENSUS --output BOUND` authenticates child walks and admits reuse only with identical prior native block history and conditioning. Otherwise it schedules parent-state reconstruction.
3. On one quiet A800, `python -m experiments.longer_run_headroom.run --inputs BOUND/PLAN.json --output NEW_RUN` enforces a clean execution checkout, pinned artifacts/runtime, 8 GiB disk reserve plus 2 GiB output allowance, 8,000 new native draft/verification rounds and 45 occupied GPU minutes (including initialization/publication/rebuild). The host stops at 44.5 minutes to reserve cleanup time. No automatic retries.

The worker extends PR34's native cache constructor. It captures the original parent ingestion boundaries and target-prefix KV, rebuilds each checkpoint's draft cache through the same native ingestion sequence, checks all neutral proposals/verdicts/commits, and executes one child block without changing the original 2,048-token cap. We retain raw output before checks. Counts of canonical prefix extension and verified repair remain separate. All computations use the existing native BF16 path.

Two parent-only classes: SHORT has exactly one consecutive correct follower and an observed incorrect second follower; LONG has at least two. Both require two available follower slots. Canonical output IDs, including valid EOS, define follower correctness. Opening rounds are descriptive only.

The predeclared analysis uses 10,000 request-cluster bootstrap draws (PCG64 seed20261112). A is exact LONG share of later starts times sampled LONG blocked-with-two-followers fraction. B is realized two-follower acceptance among verified LONG repairs. Common (j,v) standardization uses the smaller parent census count. Undefined/sparse denominators and incomplete reconstruction remain visible. Local token gains are not converted to verifier-call savings.

Run CPU checks with `python -m unittest discover -s tests -p 'test_longer_run_headroom*.py' -v`; inherited native-state constructor checks use `test_verification_transition*.py`.

Private source prompts, canonical answers, full token arrays and runtime authentication records stay outside Git. Public reports cite the actual execution commit. Native staging uses the existing qualified environment/compiler/thread settings, with no new generation warmups.
