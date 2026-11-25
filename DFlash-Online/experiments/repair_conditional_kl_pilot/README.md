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

The original November24 resource admission was **qualification only**. A one-event
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

The original production blueprint was one event per arm,128 own feedback responses,
256 presentations and128steps each;128 shared held-out requests with3timing
repeats (768scored completions plus24declaredwarmups). Main training, witness
outcomes and serving gains remain unexecuted until resource admission is changed.
Private inputs, token traces, teacher tensors and weights stay outside Git.

## One-event follow-up

The later `Codex_PAIR_One_Event_Execution_Handoff.txt` supersedes the fallback
for a **new** production attempt. Historical `prepare.py`, `launch.py`, sealed
qualification plan, and report remain unchanged. The scientific objective is
unchanged as well.

`prepare_production.py` authenticates the retained qualification and unchanged
scientific dependencies, seals fresh BASE/PAIR configurations and the same
ordered inputs, and checks real storage. It writes an explicit
`reused_passed` receipt with zero new qualification optimizer steps. A blocked
plan remains blocked; it does not create a GPU clock or repeat qualification.

`production.py` runs BASE then PAIR, each with serving on physical A and training
on physical B. Fresh released origins are compared exactly, and the qualified
disposable startup is restored before learning. After both clean exits,
evaluation uses physical A while B is idle. The existing 768 scored plus 24
warmup schedule, canonical checks, request pooling and paired bootstrap are
reused unchanged. No retry or event-two path is added.

`production_storage.py` measures actual filesystems, charges allocated blocks
once per inode, budgets phase peaks separately, and checks both the 400 GB main
reserve and any secondary reserve. `local_single` retains both feature pools on
the main volume. `local_split` requires independent durable storage for both
pools, with eight verified BASE packets retained on the main volume. An alias,
RAM filesystem, unverified offload receipt, or unadmitted route cannot launch.

The original eight-response witness now preserves each origin error position,
canonical token and winning rival. Existing readouts supply original-rival and
current-strongest-rival margins, native tie decisions, useful repairs, earlier
prefix damage and full-block progress. No additional forward is introduced.
The matched per-block table contains private token identities and stays outside
Git; witness results are training-fit diagnostics, not held-out efficacy.
