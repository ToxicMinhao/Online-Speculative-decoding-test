# Implementation review before native execution

Execution source: c504d48b05f79368e18bd9ec9c76e0e59c658205, PR40.

The ARS experiment workflow is applied inline. Separate-agent delegation was attempted repeatedly but the platform returned an agent-thread limit. This is same-author source review plus a separate arithmetic implementation planned for results; it is not an independent human or blinded review.

18 focused CPU checks pass: EOS/cap/seed handling, SHORT eligibility, fixed sample order, blocked-versus-repaired distinctions, accepted followers, signed changes, exact neutral proposals, offset-only rejection, one-round exit with original cap, terminal case, and failure preservation. 57 inherited native-state/budget/data checks also pass. GPU integration remains unqualified until actual neutral replay checks succeed.

Review confirms parent outcome separation and authenticated 128 parent walks. The sealed sample was completed before KPLUS6 file reads. Only one pair has an identical prior native walk; all255 others are reconstructed, never replaced.106full parent walks plus255neutral and255child blocks give6,895planned native rounds; normal engine initialization is separately counted inside the8,000round and45minute bounds. One physical GPU, no gradients/optimizer, no new checkpoint or serving-timing claim.

The adapter preserves original2048-token sampling cap and4096context. It rebuilds destination draft KV from recorded ingestion segments, retains target-prefix KV, and exits after exactly one native round. All15neutral proposals and native accepted/committed tokens must match archived evidence. Child emitted tokens must match canonical continuation. Failed evidence is saved before raising, affected starts are invalid or pending as appropriate, and no automatic retry/replacement is permitted.

The host admits pinned models/runtime/CPUenvironment, clean source checkout and2GiBoutput+8GiBreserve. Bothcardsmustbeidle at admission. Occupiedtimeincludesmodelinitialization,publicationsandKVreconstruction; host reserves30secondsforcleanup. Source and checkpoint postflight checks detect mutations. Ordinary K+ training/inference source remains unchanged.

Analysis additions are CPU-only and separately published; execution commit remains fixed. No performance win is required. The final conclusion must distinguish a remaining opportunity from realized repair payoff and cannot convert local token changes into saved verifier calls.
