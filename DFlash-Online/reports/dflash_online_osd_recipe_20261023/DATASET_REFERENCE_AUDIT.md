# Dataset-first OSD evidence audit

## Material Passport

October23,2026. ARS validate; status ANALYZED. Retained executed reports checked
with independent read-only evidence review. No model execution, remote operation,
setting change, dataset change or new experiment. User corrects the comparison
priority: verify GSM8K/CodeAlpaca/ShareGPT before choosing a successful reference.
This supersedes using Tulu conversation success as the main current comparator.
The earlier Tulu measurements remain valid for their own different workload.

## Strongest positive dataset/model match

The native Qwen3-8B/Qwen3-8B-DFlash-b16 OSD study used GSM8K,128-response buffers,
58events and7,424steps. On1,024disjoint held-out GSM8K requests it reduced
verification calls50,549to44,556(11.86%), with identical output token IDs. This
matches the current released model pair and dataset family, but not the eight-
response learning budget. It is not evidence for CodeAlpaca/ShareGPT or total
serving-plus-training speedup.

## Earlier all-three-dataset small-feedback controls

These use Llama-3.1-8B-Instruct/DFlash-UltraChat and qualified AdaFlash OSD-style
forward KL. Each domain has8ordinary held-out requests. Positive percentages
below mean fewer verification calls against each request's frozen baseline.

| Dataset | Frozen calls | S2:4responses/1pass/2steps | S8:4responses/4passes/8steps | L8:same asS8 at3e-5 | P8:8responses/2passes/8steps |
|---|---:|---:|---:|---:|---:|
| GSM8K |298|342(−14.77%)|319(−7.05%)|299(−0.34%)|352(−18.12%)|
| CodeAlpaca |633|634(−0.16%)|637(−0.63%)|630(+0.47%)|678(−7.11%)|
| ShareGPT |926|942(−1.73%)|945(−2.05%)|929(−0.32%)|954(−3.02%)|

S2/S8/L8 refer to two independent4-response groups per dataset; P8 pools the
eight available source responses into one trajectory per dataset. These are
not all the same experimental allocation. L8's CodeAlpaca improvement is real
in the retained count comparison but consists of only3calls on8requests; it is
not a robust substantial cross-dataset success. P8 is the closest historical
match to the eight-response/two-pass budget and did not improve any domain's
ordinary held-out aggregate.

## Positive source fitting and limited related-instance transfer

The same retained S8 checkpoints improved native source replay on all domains:
GSM8K346to233calls(+32.66%), CodeAlpaca502to311(+38.05%),
ShareGPT798to457(+42.73%). These are8source requests per dataset, already used
in training. They establish fitting ability, not future-request transfer.

A separate presealed constructed related-instance diagnostic (4new requests
per dataset, changed answers) yielded:

| Dataset | Frozen toS8calls | Reduction | Boundary |
|---|---:|---:|---|
| GSM8K |149to106|+28.86%|All4children improved; narrow constructed transfer |
| CodeAlpaca |259to281|−8.49%|All4children regressed |
| ShareGPT |412to394|+4.37%|3improved/1regressed;17of18saved calls from one child |

These children are outside training but are deliberately constructed related
instances, not ordinary independently sampled held-out requests. Do not combine
them with the ordinary-test population or call a local forward result measured
serving acceleration. This diagnostic did not include controlled timing repeats.

## Comparison decision boundary

No reviewed single small-feedback recipe establishes useful ordinary held-out
improvement across GSM8K,CodeAlpaca andShareGPT. Do not use a successful Tulu
conversation setup as if it were that missing result. For a positive reference
matching currentQwen/GSM8K, the native persistent128-buffer run is strongest;
for a same-eight-response/all-three-dataset control, P8 is relevant negative
evidence. Related-instance positives are a separate narrower category.
The new selected DFlash-Online recipe has CPU checks but no new full GPU efficacy
result; its old negative measurements must not be attributed to the revised settings.

## Sources and interpretation

- [Native Qwen/GSM8K result](../adaflash_reproduction_20260930/ONLINE_RESULT_REVIEW_20261002.md).
- [S2/S8 learning curve](../dflash_baseline_learning_curve_20261020/execution_001/final_review/REPORT.md).
- [S8 source/held-out diagnostic](../dflash_kl_native_diagnostic_20261020/execution_001/final_review/REPORT.md).
- [Lower-LR control](../dflash_lower_lr_control_20261020/execution_001/final_review/REPORT.md).
- [Eight-response P8 control](../dflash_feedback_allocation_control_20261021/execution_001/final_review/REPORT.md).
- [Related-instance diagnostic](../dflash_related_instance_transfer_20261021/execution_001/final_review/REPORT.md).

ARS11/11 interpretation categories considered: explicit population/model/budget
separation, positive and negative evidence retained, no cherry-picked best group,
no source-as-heldout substitution, no causal claim about response budget or dataset,
no small-count significance claim or timing-speedup inference, no new selection
by observed test outcomes. Historical results remain unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_recipe_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
