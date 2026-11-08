# Role-loss pilot input audit

## Material Passport

- Status: **PASS — no blocking input-contract issue found**.
- Scope: read-only review and independent recalculation of the root-authored input adapter and the prepared input package; no model, generation, optimizer, GPU or remote operation.
- Auditor: separate same-team `role_loss_core` agent. This agent authored the loss reducer but did not author the input adapter. This is not an external audit.
- Checked at: 2026-11-07 22:31 UTC. Stage 1 / project None unchanged.
- Contract: [HANDOFF.md](../HANDOFF.md), SHA256 `98c30ddb66d0aaea8cd2d06a58db9911ba644d0374bbb2b8cad2d833d94dfa62`.
- Artifact: [prepared_001](../prepared_001/). The later production seal must bind these reviewed bytes and the retained source identities.

## Findings and verification

The seven input unit tests passed. A separate Python calculation, importing no production selection, normalization, hashing or MathQA input helper, passed **20,245 assertions**. It reread the authenticated archive and retained plans, independently normalized inputs, rendered/tokenized all 29,837 MathQA training rows with the authenticated tokenizer, reconstructed source-order duplicate and panel-overlap filtering, and recomputed every hash rank. It did not use dataset answers or response outcomes in selection.

| Contract check | Result |
|---|---|
| Exact SHA256 bytes | `vfd-roleloss-pilot-v1\|{dataset}\|{split}\|{source_id}` in UTF-8, without quotes/newline; original source ID retained, tie broken by source ID |
| Evaluation selection | Both inherited plans contain 1,024 requests; each selected 256 IDs and order exactly matches independent hashing |
| Evaluation before training eligibility | Adapter determines both complete subsets before building the evaluation overlap index and before enumerating/filtering training candidates; no training result affects panel selection |
| GSM8K stream | All original fields of the first 512 requests from the 7,473-request manifest are unchanged, including IDs, tokens, order, event assignment and output cap; metadata and stop IDs are added |
| MathQA source | Exact Git blob `193fb150706468110459ddcdc52ad48e3f0b7491` from the retained revision `4f958013c164a40745a5b771dfb005df308deb4a`; `train.json` has 29,837 rows and `test.json` 2,985 |
| MathQA source-order deduplication | Ascending original row enumeration; the first valid input reserves its normalized stem/rendered prompt before subsequent rows are checked, even when that first input is later ineligible |
| MathQA input eligibility | 29,827 eligible rows; exactly 10 exclusions, all `training_context_budget`; no duplicate or evaluation-overlap exclusion occurs in this archive/panel combination |
| MathQA selected/unselected order | All 512 selected IDs, token lists and hash keys match; all 29,315 unselected eligible IDs and all ten exclusion IDs/reasons also match |
| Prompt budget | Full rendered training prompt at most 224 tokens plus output cap 2,048 fits context 2,272; selected maxima are GSM8K 183 and MathQA 200 |
| Prompt content | All 768 selected MathQA train/test prompts equal `Problem.strip() + "\n\nOptions: " + options.strip()` from the original source row; no answer/rationale/formula field is included |
| Wrappers and normalization | All 1,536 prepared train/test prompts retain the qualified single-user, reasoning-disabled Qwen wrapper; NFKC/casefold/whitespace-normalized stems and rendered/token hashes independently match |
| Train/evaluation overlap | All 1,024 training inputs are disjoint from both selected panels by dataset/source ID, normalized stem, rendered-prompt hash and token hash |
| Retained evaluation inputs | All 512 prompt-token arrays and serving fields are unchanged from the selected inherited requests |
| Canonical references | All 512 records are exact retained copies, including tokens and termination; no missing reference or new reference generation |
| Diagnostic inputs | Each diagnostic set is exactly the first 16 hash-ranked selected evaluation inputs, with the prescribed maximum of eight anchors |

Code inspection confirms `read_mathqa_train` passes only `Problem` and `options` out of the archive adapter. The selection functions do not read latency, correctness, response length or dataset answers. Canonical validity checks run after input selection and fail rather than substituting another request. The source-order and outcome-independence synthetic tests supplement the actual archive, which contains no deduplication/overlap removals under these checks.

The 224-token eligibility is a conservative input-only bound reserving the full allowed 2,048-token output. It is not filtering by observed output length. These remain previously used development panels, as the prepared metadata states.

## Reviewed identities

Code paths are relative to `DFlash-Online` in the publication checkout.

| File | SHA256 |
|---|---|
| `experiments/role_loss_pilot/data.py` | `4a1c7c806ea8d62513aae8d2ed5c0fb1aa1e82371e035b3e71387c0dabbbe9c3` |
| `tests/test_role_loss_data.py` | `dd2e5bd88b88398a1fec9a50df730ed6ccd389edd7951bd6b3f400eea851dfee` |
| MathQA archive, 7,302,821 bytes | `7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a` |
| `train.json` member, 23,388,469 bytes | `00e8919347d65dbba9289bf04ed998a6c48dbf451ca909eeb66a35f2419c2bf6` |
| `test.json` member, 2,346,754 bytes | `dfe7bc4691caf26842ccd4cf14e8f978327bab4f2989e0b076a4e6b38a9371d1` |
| GSM8K retained `PLAN.json` | `b8276f99a4922f130778a0c452ca81049a6a36980db6f5db9e35b848b8fcf55a` |
| GSM8K retained `ORIGINAL_MANIFEST.json` | `764fa339938a240259949040134581a2231de542b7c799af5a7eb85c5ea1dbd4` |
| GSM8K retained `CANONICALS.json` | `065f3fae37d49dfa448f362108ad5befbd50406934ce2cdcd1dd0e3be4c8ae6c` |
| MathQA retained `sealed/PLAN.json` | `25154b6a9b472f25861a60c8ba4fc45c26b658b99975e0418fa5d871256c1ea0` |
| MathQA retained `campaign_001/evaluation/final/CANONICAL.json` | `4d7f765ccefe4423812da4529d3145931b57bca51eeedc4974b7a7f876b5bad5` |

Source locations: GSM8K `research/dflash_adaflash_arepo_20261107/production_001/sealed`; MathQA `research/dflash_mathqa_transfer_20261104/retrieved_002/production_002`; archive `research/dflash_mathqa_transfer_20261104/data_preparation/MathQA.zip`, all under `/home/minhao2/EAGLE_SPEC`. Tokenizer snapshot `b968826d9c46dd6066d109eabc6255188de91218` was loaded locally, and both required tokenizer-file SHA256 values match the pinned PR26 identities.

| Prepared file under `prepared_001` | SHA256 |
|---|---|
| `PREPARATION.json` | `4a840a500a032c5149b461b3869c10fbe9825272dfc3bc863235d0e4f380cb58` |
| `MATHQA_EXCLUSIONS.json` | `68b59ea74f4e6b0f99f55b689179356dcd4a695c82da56e44ef0dbcebd1ee25d` |
| `gsm8k/TRAINING.json` | `1acc321ff0696e47025a1bc3e5109c84cb77715dfa1a6919e7d2c58b610c49e9` |
| `gsm8k/PANEL.json` | `85f9e163445616c2cd8b37895f068d42e929a6d7f746f2ccd72b9bbb9b00d9cd` |
| `gsm8k/CANONICALS.json` | `e8466d2a25ad4df468c9846bfa05c13386b92f230d09ce52eafad942bfc9d680` |
| `gsm8k/DIAGNOSTIC_INPUTS.json` | `17b17e5ae421bf436c08dab18e8e29135e3240d1054f34d66e93f397e4bebf7d` |
| `mathqa/TRAINING.json` | `15a6b3d09ef9a10f855ac0294ee1672c2c18dbd17fa22a8f352819313dbd7392` |
| `mathqa/PANEL.json` | `b00377524069f81ac84ee52d00ec695a9e0db010e3d43cba91ec5113d65dd104` |
| `mathqa/CANONICALS.json` | `f3554a73356aede23eb1ff270b8fa7e593630e99ea8ba2a4820555f01934c89f` |
| `mathqa/DIAGNOSTIC_INPUTS.json` | `d50f91c6907d3d40de102c3b468e097652ba2074cbf91c2c8d56e63669473168` |

No adapter or test edits were needed. This audit qualifies input selection and preserved reference identity; it does not qualify the model runtime, training, canonical-context diagnostic execution, timing, or endpoint efficacy.
