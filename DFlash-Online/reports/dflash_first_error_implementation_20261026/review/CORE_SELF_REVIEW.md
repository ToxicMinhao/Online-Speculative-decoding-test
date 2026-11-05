# C/K/A core implementation review

ARS experiment-agent, CPU implementation stage. This is the implementing agent's
self-review and reproducible archived/current comparison, not an independent
external audit. No GPU qualification, scientific training or generation occurred.

No confirmed core defect was found in this review. The implementation retains
the existing full-block draft forward, same-position saved canonical labels,
predecessor-aligned teacher features, original validity/decay weights, optimizer
and accumulation code. The K/A mask is computed from detached current logits
independently in each projection block on every presentation. It excludes the
seed, includes the first mismatch, includes a valid canonical EOS and prevents a
gap or terminal boundary from reopening the prefix. C uses all existing valid
proposal weights.

C/A compute `-log_softmax(logits).gather(canonical_label)` in the configured
qualified BF16 probability precision. They never enter legacy `hard_ce` or
construct/project a teacher/reference. K uses the unchanged O teacher KL row
calculation. Numerators and selected weights are summed over projection chunks
before one response-level `1e-6` denominator epsilon. The existing response-mean
adapter and TrainerCore preserve equal response averaging and accumulation.

Counts and histograms are returned as detached additive chunk terms, while
checkpoint recomputation only repeats real compute and timing work. Event totals
sum optimizer-boundary rows once; each new-arm row labels whether its metrics
describe one presentation or a whole accumulation window. Available counts use
the original validity weights. No-error/empty counts and first-error histograms
describe the contiguous valid chain ending at EOS/padding, even for C. Thus a
synthetic invalid gap can leave later positions counted as available while
ineligible for the K/A continuation. New resume contracts explicitly identify
loss, current-forward selection, probability arithmetic, canonical labels,
EOS IDs and absence of a reference. Old objective contracts remain unchanged.

The archived/current comparator imports the October 26 R execution source under
a separate package namespace. It compares O/U/P/R and legacy hard-CE/projected
losses, accuracy, numerical metrics, all 14 gradient tensors, RNG and registered
parameter identities exactly. Synthetic nonempty U/P/R prefixes separately test
active auxiliary paths; the synthetic R reference softplus baseline is explicitly
zero for both versions. These fixtures do not claim that the natural frozen
reference selected those positions.

- [Comparator](compare_old_new.py)
- [JSON receipt](OLD_NEW_COMPATIBILITY.json)
- [Execution log](OLD_NEW_COMPATIBILITY.log)

The comparison passed all 72 checks over nine old/new pairs, with 18 tiny CPU
backwards, zero optimizer steps, zero generations and CUDA uninitialized.
Existing focused O/R objective/recipe/trainer tests additionally passed 26 tests.
The first attempted invocation used an unrelated `envs/dflash` Python missing
`pydantic`; eight objective tests passed and three imports failed. The actual
qualified environment is `/home/minhao2/.venvs/dflash-specforge/bin/python`.

These results cover eager CPU arithmetic and code compatibility. They do not
establish native GPU/Flex execution, serving correctness, loss effectiveness or
end-to-end benefit; those require the separately bounded GPU qualification and
campaign after this implementation review is returned.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_implementation_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
