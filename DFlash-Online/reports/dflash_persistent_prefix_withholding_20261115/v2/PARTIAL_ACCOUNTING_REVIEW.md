# Retained qualification failure: independent accounting review

Execution `e032cbc1fba90a7a6aa6a3460bc0a8c954810497` stopped in the ALL-versus-native control. The experiment has not reached an E trial, production training, feedback generation or serving comparison. The original report correctly labels this incomplete and makes no efficacy claim.

The durable ledgers show **two completed disposable Adam applications, six response forwards and six backwards, one training initialization, and zero complete-response generations**. Two of the six forwards/backwards belong to separately counted zero-Adam startup; four belong to the two control trials. There are no unknown or failed physical operations. The qualification comparison failed after completed operations. All ten production tasks remain unattempted, and all **1,640 planned generations** remain unattempted.

The two trials have identical response/anchor identities, forward observations, pre-clipping gradients and RNG. Both trainer counters report step 769. However, the retained exact FP32 scalar identities show Adam step 769 for native and 770 for ALL, for all 58 parameters. Both moment tensors agree across all 58 parameters, while 36 of 58 parameter tensors differ. These records identify a concrete optimizer-state discrepancy; they do not support dismissing the failure as BF16 gradient sensitivity. Source-level restoration diagnosis remains separate. Compact hashes establish inequality but do not provide elementwise weight-error magnitudes.

Recorded occupied-card time is **89.015991 seconds**; launch wall time is **148.703789 seconds**. These nested clocks must not be added. The independent final-audit tool passed **990 assertions with zero errors** under `--partial`, and deliberately computed no efficacy result.

The earlier attempt stopped at CPU environment-version checking, before CUDA initialization or model work; its evidence remains preserved separately. No raw report was rewritten, no scientific setting changed, and this review performed no model operation or retry. A repaired control still requires native qualification before any campaign outcome can be interpreted.
