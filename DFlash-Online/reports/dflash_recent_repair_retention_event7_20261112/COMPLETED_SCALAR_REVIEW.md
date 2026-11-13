# Completed scalar evidence review

Scope: read-only review of the completed `early_report/final_report` package.
No remote operations, model execution, retesting, tuning or partial-result
selection. Full private-artifact review awaits the retrieval verification
receipt. The parent reports a separate raw audit with 814,215 passing checks;
this note does not relabel that as an independently repeated native experiment.

## Decision

The reported decision, **prespecified useful size unsupported on this fork**,
follows the original handoff. RECENT's accepted draft tokens per verification
are 5.641520 versus NATIVE's 5.659532: relative gain −0.3183%, 95% interval
[−0.7478%, +0.1044%]. The upper bound is below the prespecified +0.50% screen.
This is not a significant-harm claim or evidence of equivalence.

The fixed comparison comprises 256 paired requests, two sweeps and all three
arms: 1,536 valid executions, zero invalid/unknown/unattempted executions.
Counts are used once per request; timing is averaged across repeats. All 22
reported uncertainty quantities pass the independent statistics reduction
(27,486 checks; zero failures; PCG64 seed 1707, 10,000 whole-request draws).

RECENT saves 107 calls and adds 136 relative to NATIVE, net 29 additional calls
(11,055 versus 11,026), with 80 wins, 94 losses and 82 ties. Its fresh time
reduction is −0.2176%, interval [−0.6016%, +0.1615%]. Neither timing improvement
nor a clear slowdown is demonstrated. RECENT versus CONTROL is also unresolved:
accepted-length gain −0.0671%, interval [−0.5067%, +0.3726%]. The placement
screen passed, which makes the control usable; it does not establish a
placement benefit.

## Mechanism evidence

Across 2,775 fixed starts in the sealed 64-request population, all arms share
431 eligible repair slots and 2,410 genuine K6 blockers. RECENT has 123 exact
relapses and 275 accept-through outcomes versus NATIVE's 121 and 278, and
CONTROL's 118 and 281. The desired retention direction is absent in the point
estimates, with request-cluster uncertainty retained in the machine-readable
report.

RECENT acquires 1,356 and loses 1,361 prefix occurrences (net −5), compared with
NATIVE's 1,327 acquired and 1,270 lost (net +57). Thus 29 additional acquired
occurrences coexist with 91 additional losses at these fixed starts. This is
local diagnostic accounting, not an explanation of a percentage of serving
time, nor proof that a particular protected training slot caused those losses.

The auxiliary mechanism was active: RECENT fired in 160 presentations and 107
of 128 optimizer windows. Its first natural firing changes 297,041 BF16 weight
entries relative to the same-step native update; the final endpoint differs
from NATIVE in 296,587,104 entries. These establish a realized intervention,
not usefulness. A weak-dose causal diagnosis is not established merely from
the sparse assigned weight.

## Accounting and reporting details

- Scientific work is 384 Adam steps and 768 presentations across three arms.
  The completed training is reused during evaluation recovery, not repeated.
- Preserve the separately counted three qualification Adam steps and four
  first-firing replay Adam steps, plus startup/reference preparation and the
  two new canonical-reference generations. These are outside scientific
  training and the 1,536 timed evaluations.
- Preserve the failed pre-generation attempt: 24.235622 occupied GPU seconds,
  one failed parent-walk wrapper, 51 startup draft blocks and 48 startup target
  calls. No parent response generation had begun in that failed attempt.
- Combined execution wall is 6,258.991326 seconds, excluding repair/user-wait
  gaps. Allocated phase windows total 5,997.623071 seconds, or 1.666006
  card-hours. The distinction from kernel-busy time is correctly disclosed.
- Peak reported allocator usage is 19.105625 GiB allocated and 24.039063 GiB
  reserved; these should not be renamed total physical VRAM usage.
- Main training is 339.371692 seconds for RECENT versus 333.391153 for NATIVE:
  +5.980539 seconds (+1.793850%). This is a measured process difference for
  this run, including the documented bounded preclip-copy work, not an isolated
  intrinsic-kernel overhead estimate. Diagnostic state capture/replays and
  persistence/export remain separately reported.
- Reused feedback, teacher preparation and historical label replays are not
  newly timed. Their required future cost is disclosed; no training-inclusive
  deployment speedup can be inferred.

One presentational addition is needed to fully satisfy the handoff: publish
explicit weighted live-reachable and firing fractions, with denominators.
The retained numerators already support them:

| Arm | Reachable / assigned weight | Firing / assigned weight | Firing / reachable weight |
|---|---:|---:|---:|
| RECENT | 96.369674% | 22.364923% | 23.207428% |
| CONTROL | 96.460201% | 20.566653% | 21.321387% |

Per-pass fractions should use each pass's own retained weights. The separate
panel-publication addition supplies candidate counts, exclusions and selection
hashes missing from the original compact report. Publish a redacted raw-audit
receipt; keep absolute artifact paths in the private copy.

No scalar correctness blocker was found. The fixed pilot does not support an
upgrade over native K+ and does not trigger a new experiment. Uncertainty about
other doses, seeds or trajectories remains outside this completed screen.

Authorship: same research team; this reviewer contributed preparation/matching
and a separate raw reduction. This is not independent GPU replication.
