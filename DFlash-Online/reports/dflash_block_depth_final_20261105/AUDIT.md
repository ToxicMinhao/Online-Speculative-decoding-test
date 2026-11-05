# Frozen-depth completed-record audit

Verification Status: ANALYZED

Execution `0f704167b268672d980fabe704898abf9b134c39`; checks 107718; errors 0; complete supported True.

{
  "ADAPTIVE_vs_FIVE": {
    "point_pct": -0.8084014269725825,
    "ci95_pct": [
      -0.9915198580808954,
      -0.6417508389591786
    ]
  },
  "ADAPTIVE_vs_THREE": {
    "point_pct": 55.84822998032757,
    "ci95_pct": [
      54.86723453442621,
      57.05507255059953
    ]
  },
  "THREE_vs_FIVE": {
    "point_pct": -128.32244637543639,
    "ci95_pct": [
      -134.7444913729219,
      -123.37197302418019
    ]
  }
}

Standalone NumPy/stdlib reconstruction; author also implemented gate/endpoint/data/plan. Not blinded and not an independent GPU repetition. Retired feature tensors and full GPU state are not re-executed.

Complete-request timing is primary. Native decode subcounter is secondary. FINAL is fit-disjoint but historically exposed. Fixed checkpoint, one gate, one dataset; no online recalibration or training-inclusive speedup inference.
