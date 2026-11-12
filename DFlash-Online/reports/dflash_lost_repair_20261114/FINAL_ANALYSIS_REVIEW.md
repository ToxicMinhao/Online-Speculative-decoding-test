# Lost-repair final arithmetic and interpretation review

ARS experiment-agent / retained-evidence validation. Scientific execution: `29b6a15c5f5f16135663ea61e3d31f21b3f1d465`. No model execution, new data, changed settings or scientific source edits were performed for this review.

**The experiment completed correctly, but the prespecified feasibility gate failed its weight criterion.** This is a completed negative screen, not a qualification failure.

## Arithmetic verification

A separate stdlib reduction, without importing the production analysis helpers, reproduced **202,150 scalar and table checks** with no discrepancies. It checked native census labels, depth/lost-prefix histograms, request totals, the exposure funnel, fixed-seed nearest-margin matching without replacement, authoritative native weight denominators, all 64,564 published sampled-occurrence weights, matched-pair rows and the gate. This reviewer also authored reporting code: the computation is independently implemented, but authorship is not independent. See `FINAL_SCALAR_AUDIT.json` and `final_scalar_audit.py`.

The separately authored retained `delivery/RAW_AUDIT.json` reports **1,031,607 checks, zero errors and zero missing evidence**, covering raw native records and public reductions. This review did not rerun those native forwards.

| Quantity | Verified result |
|---|---:|
| Requests / natural starts | 128 / 4,877 |
| Genuine K+6 rejections | 4,252 |
| Lost repairs H | 252; 5.1671% of starts; 5.9266% of genuine rejections |
| Requests containing H | 105/128; largest request contributes 10/252 H |
| Persistent errors M / converse cases | 2,147 / 341 |
| Previously attained then lost prefix occurrences | 764 |
| Native H first-error mass before matching | 315.5024515539408 |
| Full native first-error mass | 27,646.563011169434 |
| Full selected loss mass | 288,152.9246826172 |
| Projected matched mass | 40.617696069180965 |

H accounts for **1.14120% of native first-error mass** and **0.109491% of all selected loss mass**, before matching. These are distinct denominators.

| Exposure stage | Distinct H anchors | Requests | Scheduled occurrences |
|---|---:|---:|---:|
| Native H census | 252 | 105 | Not a presentation count |
| Appears in schedule | 252 | 105 | 503 |
| Training first mismatch agrees on j | 240 | 104 | 479 |
| Comparable M available in the required stratum | 119 | 69 | 237 |
| Matched without replacement | 115 | 69 | 229 |

The support gates use **115 distinct anchors** and **69 distinct requests**, not 229 repeated-presentation pairs. The weight gate sums every matched scheduled occurrence and divides by the entire event's native first-error mass. Its value is **0.1469177057%**, below **0.25%**. The 32-anchor and 16-request gates both pass.

## Supported interpretation

The proposed phenomenon is present across many requests. It is **not explained by these historical H starts being absent from the sampled training anchors**: all 252 distinct H starts appear. This does not establish that their allocated weight is sufficient or that learning on them would transfer. Twelve distinct starts lose the same-j correspondence under frozen training conditioning; that is separate from absence in the schedule.

The largest coverage reduction is the required H/M comparison: 240 same-j H anchors become 119 with an eligible comparator and 115 matched anchors. Of 479 same-j H occurrences, 242 lack a comparator in the same presentation, response, depth and priority class; eight more remain unmatched because M cannot be reused. Matching therefore leaves insufficient projected mass under the unchanged screen. This identifies limited support for the prescribed matched allocation, not failure of every possible recovery mechanism.

Nearest-available matching also does not achieve close margin balance: mean H wrong-minus-canonical margin is **0.883734**, mean M margin **1.696507**, and mean absolute pair difference **1.107533**. These descriptive imbalances remain visible; they should not be described as an exactly difficulty-matched control.

The 341 converse cases must not be subtracted from 252 H cases to claim net decoding improvement. Both are fixed-start historical measurements. Likewise, the 764 lost prefix occurrences are not a measured complete-request call or latency penalty, nor a demonstrated recoverable gain.

## Completion and limits

The retained receipts report 128 captures, 4,877 neutral replays, 9,754 historical replays and 256 read-only response forwards, with no failed or unknown work. The full draft-block total is 19,559: 19,508 live capture/replay blocks plus 51 startup blocks; startup is a subset, not additional work. Training used 8,170 projection chunks and zero backwards or optimizer steps. Aggregate occupied-GPU time was **2,853.5036 seconds (47.5584 minutes)**, below the four-hour ceiling. Publication and retrieval are separate from this occupied-GPU measurement.

The drafted interpretation in `INTERPRETATION_REVIEW.md` is supported with the precise wording **“no missing-anchor exposure gap on this batch”**. Avoid the broader claim “no exposure limitation”: the effective mass is small, and its adequacy was not tested. The proper endpoint is publication of the completed failed gate and a stop, with no implemented mechanism, adjusted threshold, new matching rule or additional experiment inferred from these data.

Sources: `delivery/retained/campaign_001/{census/CENSUS.json,training/RESULT.json,RESULT.json,final_report/}`; `delivery/retained/delivery/RAW_AUDIT.json`; `reviews/FINAL_SCALAR_AUDIT.json`.
