# Role-loss pilot implementation review

Status: implementation and CPU validation; native qualification and efficacy results are not yet available.

The user authorized the fixed KPLUS/MIX/ROLE pilot on November 7, 2026. The isolated reducer preserves ordinary K+ selection, priorities, redistribution, native BF16 row arithmetic and response averaging. MIX applies 0.8 teacher-argmax CE plus 0.2 released-repository clipped reverse KL to selected positions. ROLE retains forward KL on the matching prefix and applies the mixture only to the first error. The full response denominator D sets the clipping scale; the selected denominator Z normalizes the loss. ROLE with no first errors returns the native K+ graph.

The six trajectories initialize independently from released weights and authenticated seed-17 fresh optimizer/RNG. Each runs four own-feedback events. The inherited 7,424-step configuration horizon remains unchanged; an experiment wrapper stops each trajectory after 512 Adam steps. Both event-2 and event-4 exports are retained. All training finishes before the contemporaneous, capture-disabled evaluation phase; event-2 results cannot affect training or endpoint selection.

Input selection was independently recomputed: 20,245 assertions pass across the fixed GSM8K first 512 requests, all 29,837 MathQA training rows, both 256-request hash-selected panels, exclusions and identities. All 512 canonical native records are available and their complete traces validate against the exact panel inputs. No new canonical generation is planned. MathQA training has ten input-context exclusions; selection never uses answers or outcomes.

The implementation separates source fitting, fixed-context diagnostic changes and actual native serving results. A dedicated worker exposes retained target-feature reconstruction without changing serving methods or installing depth routing. The fixed 32 diagnostic contexts are never used for training. Native target reconstruction forwards and terminal preparation are counted separately.

Integration review corrected an evaluation-boundary receipt mismatch, added retained-export identity checks, repaired launch test names, and strengthened disposable decoded-state restoration. These are execution/integrity fixes; they change no objective, setting or tolerance. Six disposable qualification steps are bounded separately from the 3,072 main Adam steps.

Storage admission includes all six final recovery states, twelve selected exports, active atomic overlap, diagnostic tensors, raw records, compiler/workspace allowances and an 8 GiB reserve. Historical evidence is preserved. Only validated redundant/intermediate artifacts of this new run use rolling retirement; receipts record lost replay/resume capability. The conservative maximum is approximately 149.073 GB, pending fresh admission immediately before launch.

Review provenance: separate agents in the same assistant team checked data and integration. The loss implementer also audited the dataset; the evaluator implementer audited campaign integration. This is useful separated-work review, not an independent laboratory replication. CPU checks cannot establish native GPU correctness or performance. The final report must identify the published execution commit, qualification work, every valid/invalid/unattempted measurement, costs, both dataset endpoints, and a separately calculated arithmetic audit.

Scope remains fixed: 3,072 feedback responses, 6,144 training presentations, 3,072 main Adam steps, 4,608 timed completions, at most six disposable Adam steps, 224 draft diagnostic response forwards and 32 target response preparations. No tuning, extra arms or full-58-event extension is authorized. A null or harmful valid outcome completes the pilot.
