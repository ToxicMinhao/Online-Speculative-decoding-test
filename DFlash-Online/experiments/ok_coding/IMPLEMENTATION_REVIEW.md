## Material Passport

- Origin: ARS experiment-agent, run/validate workflow, October 30, 2026.
- Scope: additive experiment implementation and CPU review; GPU execution has not occurred.
- Baseline publication: `8d5d4185976fc8d78b7d55a19928d86c7cb86e95`.

This driver changes experiment orchestration only. The O/K objectives, AdamW implementation, training arithmetic, inference and upstream code are unchanged. It adds CodeAlpaca source/history preparation, true step-zero state gates, two-seed/four-event execution, contemporaneous F/O/K evaluation, raw-record reconstruction, storage admission and complete-boundary recovery.

The selected source bytes contain 20,022 rows. Historical exclusions leave 13,991 training and 334 evaluation requests; 512 and 128 are sealed with the specified independent selection seeds. Source solution columns are never read. Root separately reproduced the ordered selection, all640 exact prompt texts/token IDs, partition disjointness and4,608 evaluation slots. The preparation author scanned the retained history; this full history scan was not independently repeated by root.

Forty focused CPU tests pass, including tiny DFlash/native-BF16 fixtures and a complete synthetic4,608-row evaluator/report test. Tests cover genuine empty Adam, seed29, natural event1 order/anchors, own0..4 restoration, frozen references across four cells, included gates, raw-record tampering, pending/invalid/unknown work, balanced order, paired-request bootstrap and storage overlap/retirement. No full-model GPU qualification or experimental update was performed. CPU fixture forwards/steps are test work, not scientific campaign work.

Review roles were separated within the same agent team: a preparation author reviewed the campaign/evaluation/resource code; the training author independently examined storage; root reviewed integration and independently recalculated the data selection. These are same-team reviews, not an external review or independent GPU reproduction. A resume-source admission gap found during review was repaired: explicit recovery now rechecks the original source/model/runtime/config seals and resources before loading a backend. Interrupted partial work remains a blocker.

The fixed campaign remains storage-blocked. Its conservative complete-retention bound is366,710,105,681 bytes, versus approximately150.40GB currently free on A800. No reviewed sufficient disposable artifact set was identified, and no historical artifact was deleted. See the separately published preparation/admission report for exact source revision, measured capacity and remaining work. The implementation does not relax settings, retention, reserve or outcomes to admit a partial run.

Published implementation and actual CPU admission source: `5cf1c7e2e7c4a4d08d2225b58c56488578112daf`. No GPU experiment occurred.
