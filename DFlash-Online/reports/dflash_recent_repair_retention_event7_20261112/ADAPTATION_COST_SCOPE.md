# Adaptation-cost scope

Scientific execution commit: `562f63fdff6955d5ffcf6086bf9b2f8f948dba8b`.

This event reuses PR42's K6 feedback responses, teacher packets, and retained K5/K6 verification trajectories. Current costs therefore do not newly measure feedback collection, teacher preparation, or the historical work that supplied the repair labels. Those inputs are reused artifacts, not evidence that producing the labels is free in an online deployment.

New frozen-parent margin preparation, matching, scientific training, checkpoint persistence, and the separately identified qualification/counterfactual work are measured in this run. A complete future online implementation would also need to account for obtaining prior/current prediction evidence and maintaining its state. This pilot does not establish training-inclusive or end-to-end online speedup.

Training and qualification peak allocator counters are reported where retained. Any native-serving memory observations should be labeled by their actual coverage; do not present a training-only maximum as a continuously measured global serving peak.
