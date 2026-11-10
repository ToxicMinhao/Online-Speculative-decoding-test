# Reviewed numerical-test revision

The user adopted the reviewer's revised contract on November 9 UTC. This changes
which native calculations the qualification compares. It changes neither the
FSG algorithm nor its numerical tolerances.

- Production remains native BF16, eight blocks per vocabulary-head chunk, with
  the actual final partial chunk. Ordinary K+, optimizer and serving are unchanged.
- `same_layout_oracle.py` independently calculates prefix membership, weights,
  centered sensitivity, output energy and a complete-block parameter VJP. It
  does not call the production formula helpers. The native fixture uses the first
  nine retained anchors (one full eight-block chunk and one partial block).
- The original per-coordinate criterion remains `abs(error) <= 1e-5 +
  0.016*abs(reference)`, with global relative L2 at most 0.016. Existing independent
  high-precision CPU mathematical tests remain required.
- Original cross-layout comparisons remain **failed under their original
  criterion**, retained as BF16 layout-sensitivity evidence. They are not passes
  and are not an independent mathematical oracle.
- Exact ordinary/zero-strength behavior, restored scientific state, finite
  arithmetic, SCALAR own-state construction, actual BF16 norm/KL decisions,
  recovery/export and native output checks remain required.

`resume_qualification.py` preserves the prior cumulative work: three Adam
applications, six ordinary F/B, eighteen additional forwards and twelve
additional backwards. It performs the five remaining Adam cases. The saved GSM
ordinary endpoint and established FSG native fallback are reused; its old failed
oracle is replaced without repeating that optimizer application. The separately
user-authorized numerical investigations remain separately charged.

The twenty remaining auxiliary backwards consist of two GSM ordinary-gradient
recapture/startup, two MathQA startup, eight remaining filter-hook backwards,
four independent-oracle backwards and four expanded-resource backwards. One
resource-only test exercises two responses at 512 anchors and the qualified
2,272-token training limit, using explicitly synthetic masked context. It is
not feedback or adaptation evidence. Serving remains eight included checks.

The norm and KL guards stay strict. Finite rejection with exact native fallback
is valid execution. Record nontrivial accepted FSG corrections among the
representative native cases. If all fall back, stop before the main pilot and
report that the mechanism was not applied in those cases. Do not relax a guard
or tune another fixture to obtain an accepted update.

The original events-5–6 pilot is conditional on complete qualification, required
integration and a separate actual peak-storage admission. No partial integration
or maximum-shape check alone authorizes its launch.
