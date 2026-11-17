# MathQA fixed learning-rate transfer: preparation

## Material passport

ARS experiment-agent run workflow; fixed user-adopted November16 handoff.
This document records implementation/readiness, not experimental efficacy.
The final execution commit is sealed in the run's SOURCE.json and PLAN.json.
Private inputs, answers, teacher tensors and recovery payloads remain outside Git.

## Question and comparison

Does a fixed4% reduction of the native K+ learning rate transfer to MathQA?
N uses3e-4; S uses2.88e-4. Both restore the same complete PR32 MathQA KPLUS4,
continue their own Adam/RNG trajectories through events5and6, and finish at768
Adam steps/1,536 presentations. All native K+ arithmetic and other settings remain.

The scalar is pinned as the literal decimal2.88e-4 across configuration, recipe
and live optimizer groups. In Python float64 this differs by one ULP from
computing3e-4×0.96; the literal is fixed before any performance observation.
No parameter interpolation or gradient scaling is used.

## Prerequisites resolved

- Original recovery:6,297,245,019bytes, SHA256
  `5a21009af60e05a180140e6f582a0a5ab24ce21bbe68bcf7407078c55d23767e`.
- Original exported weights:2,097,259,104bytes, SHA256
  `931e46620d8fc930ea0c22fcdf6af30dd16e16445f8141474ecfa4e48746dff7`.
- Both physical A800s idle at the latest readiness snapshot;632.63GB free.
- Historical N reuse is disabled: the exact prepared E5/E6 feature pools and
  native E5 recovery/export were retired, not archived. Their compact results
  remain. The authorized fresh N/S fallback is used; no reconstruction of the
  four original events is attempted.
- Historical source/configuration pins were compared. Nine core native files
  and the SpecForge/DFlash training/inference trees remain byte-identical.
  Optional objective branches added later remain disabled and require native
  equivalence qualification. Training and serving use their separate pinned
  environments; their Torch versions must not be conflated.

## Inputs and costs

The exposure audit found1,960 eligible MathQA test rows after excluding1,025
previously used rows. It checked the original512 adaptation inputs, the exact256
continuation inputs, the retained earlier exposure inventory and subsequent
project request carriers. Exactly256 scored inputs and two warmups were selected
by the prescribed input-only rank before generation. This is freshness relative
to enumerated retained project inventories, not universal unknown exposure.

The DATA_SEAL SHA256 is
`058305f57b3e8a5cf914cd8c45c51c6ce645b50fbf73df4f093bef07452c2a59`.
All258 origin correctness references are new, separately budgeted work and will
be sealed before replay. Both arms share the common origin-generated E5 pool;
E6 feedback is independently collected from each arm's own E5 checkpoint.

Conservative new peak is113.634GB against the140GB cap, requiring513.634GB free
including the400GB reserve. Four new persistent boundaries and all unique
prepared features remain. Only newly duplicated capture/terminal tensors may be
retired after the existing exact feature-equivalence checks. No historical
checkpoints or unique old results are removed.

Initial estimate is approximately1–2hours after prerequisites, with measured
admission after qualification. Hard caps remain four elapsed hours and four
summed occupied-card hours from the first model load. The same clocks and work
ledger retain all failed or discarded work. CPU engineering/staging time is
reported separately.

## Required gates and interpretation

The assembled package is validated on CPU, including complete state/export and
actual runtime imports. Native qualification checks loss/gradient/clipping/
Adam/state/RNG identity, actual S rate and own-state replay on both GPUs. Only
numerical cross-card mismatch with passed local checks permits the handoff's
same-card fallback; identity failures stop. Final timing is on one quiet GPU.

The fixed endpoint evaluation has1,536 scored generations and48 warmups.
All outputs/termination must match the frozen origin references and each arm's
counts must agree across repeats. Primary intervals use the prescribed common
10,000-draw paired-request bootstrap. All favorable, null and harmful outcomes
will be reported; no tuning or additional event follows.

No efficacy conclusion follows from CPU tests or qualification. Native K+
remains primary, C remains closed, and project stage/direction are unchanged.

## Implementation validation

The integrated CPU suite passed64 tests in57.790s. The inherited staging suite
passed19 tests in0.567s after the narrow input-reader extension. Additional
source review checked restoration, common/own feedback, cross-card fallback,
reference sealing, failure-saving order and work accounting. Earlier CPU
fixture-transition and mistyped test-discovery failures are retained privately;
none consumed model/GPU work or changed the scientific protocol.

Native qualification and final results are not yet available at this preparation
revision. Passing CPU checks does not establish GPU correctness or efficacy.
