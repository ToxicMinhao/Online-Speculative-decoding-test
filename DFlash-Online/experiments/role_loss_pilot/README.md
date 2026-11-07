# K+ role-loss early-adaptation pilot

This isolated experiment compares unchanged KPLUS, uniform selected-position MIX,
and prefix/error ROLE on independent GSM8K and MathQA trajectories. Each of the
six trajectories freshly initializes released weights, native BF16 Adam and
seed-17 RNG, collects its own 512 responses, and stops after four events/512
updates. The inherited constant-LR configuration retains its 7,424-step horizon;
the pilot wrapper enforces the four-event scope without modifying old guards.

Before any production response, CPU expression/oracle tests and the sealed native
qualification must pass. Qualification restores the authenticated released state
for each of two response pairs per arm. It permits exactly six disposable Adam
steps and twelve response forward/backwards, with no startup response passes.
Loss/gradient finiteness precedes Adam, and full temporary decoded-state equality
and restoration are checked without additional model calls. No qualification
state enters production. The six-record retained inventory is authenticated; only
its first four responses are presented, twice per arm in two distinct pairs.

All six training workers run serially in fresh processes. Target features for
16 fixed canonical contexts per dataset are prepared once, then reused for one
shared initial snapshot and events 2/4 for every arm. Native reconstruction
prefill/verification/terminal forwards have their own explicit counts; these are
32 response preparations, not a claim of 32 physical target forwards.

After training, all event-2 and event-4 exports are evaluated against the sealed
256-request panel per dataset, once and twice respectively. Evaluation ordering
is sealed before execution. Event-2 outcomes cannot select a checkpoint or change
settings. Eight inherited capture/publication/reload correctness completions are
charged separately, four on each dataset's KPLUS trajectory.

The storage plan takes the maximum of training and evaluation phase peaks. It
includes all six initial states, six complete final recovery states, 12 selected
exports, active atomic checkpoint overlap, diagnostic feature storage, worst-case
compact responses, compilation/workspace and an 8 GiB reserve. Within this run,
prepared training features and superseded optimizer states are retired only after
exact successor/publication checks. Event-2 exports survive; event-1/3 exports
are retired on the following validated boundary. Historical artifacts are intact.
Every retirement records exact paths, preserved evidence and lost replay/resume
capability; retired tensors are not described as archived.

`run_campaign.sh ROOT` runs CPU checks, qualification preflight, one bounded
qualification, production admission, the campaign, report and postflight. It
records failures and never retries automatically. The final result is an early
adaptation comparison on reused development panels, not a 58-event ranking or
full AdaFlash reproduction. Stop after this pilot.
