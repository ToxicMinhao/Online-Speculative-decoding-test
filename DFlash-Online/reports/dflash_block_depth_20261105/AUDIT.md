# K+ depth qualification/pricing evidence audit

Execution commit: `9150cc3b20714153b8e4fb3d70e8a32861c5921f`.
Checks: 11955; audit discrepancies: 0.
Complete native qualification supported: True.

Generation accounting: {"planned": 32, "returned": 32, "valid": 32, "invalid": 0, "unknown": 0, "unattempted": 0}
Returned timed route measurements: 2048

Economic result: {"point_pct": 2.6389343328842663, "ci95_pct": [2.5449762367851214, 2.7034728691126215], "optimistic_saved_ms": 245.29664468718693, "native_request_ms": 9295.291725546122, "continuation_delta_ms": 0.5176008853595704, "decision": "continue_to_readout_fit"}

Separate CPU calculations without experiment imports; same-team author also implemented data/pricing/qualification, not a blinded review or independent GPU replication

No economic conclusion without complete native qualification. A pricing stop is a same-progress skip-cost budget decision, not proof that better proposals cannot compensate. No final endpoint or fitting is authorized here.
