# Repair-conditional KL pilot

BASE is the full qualified C+K+ objective. PAIR adds conditional teacher-to-draft
KL on each current first-error token and its detached native winning rival,
weighted by `(1-mu)` and the same final redistributed K+ repair coefficient.
All other learning and serving settings remain the PR56 recipe.

Start from `online_feedback/repair_conditional_kl.py`, then its three call sites
in `online_feedback/objective.py`. `training.py` keeps the native event trainer;
`configuration.py` preserves the inherited constant-LR horizon. `witness.py`
compares fixed blocks while preserving training RNG. `evaluation.py` is a true
two-arm evaluator with retained frozen-F canonicals; `analysis.py` averages
repeats within requests and uses a paired whole-request bootstrap.

The November24 resource admission is **qualification only**. A one-event
production screen was considered before GPU work, but its full-cap disk peak
requires52.791GB on A800 while only42.074GB was available above the400GB reserve.
Measured transfer was26MB/128.89s; offloading the expected3.83GB first-arm feature
pool would itself take over5hours. No old artifact deletion or relaxed limit is
used. `prepare` seals the exact source subset and blocker; `launch` executes only
the two disposable update/export/reload qualifications. Production is explicitly
blocked in the plan and runner. This follows the handoff's resource fallback,
not an early stop selected by performance.

Qualification counts: two Adam steps; four learning presentations/forwards/
backwards; four separate startup forwards/backwards with zero Adam; four
feedback generations, four teacher preparations and four canonical/reload
checks. Runtime initialization and graph captures are counted by the retained
native instrumentation. All failures remain recorded; no automatic retries.

The production blueprint remains one event per arm,128 own feedback responses,
256 presentations and128steps each;128 shared held-out requests with3timing
repeats (768scored completions plus24declaredwarmups). Main training, witness
outcomes and serving gains remain unexecuted until resource admission is changed.
Private inputs, token traces, teacher tensors and weights stay outside Git.
