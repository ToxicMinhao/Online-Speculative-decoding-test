# Executed configuration and source identities

Scientific execution: `7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a`, clean at launch. Configuration files here match the sealed remote bytes; their identities and the private full-plan/source-seal identities are in `SOURCE_IDENTITY.json`.

`config_AREPO.yaml` preserves the common trainer configuration. `online_feedback.responses_per_event=128` and `passes_per_event=2` define the actual online schedule; the inherited static `num_epochs` and hidden-state directory are not a second offline training run. Each event receives its own freshly prepared response features.

The base model wrapper retains the `osd` plumbing identifier for all-valid-position batching, but `experiments/adaflash_arepo/training.py` installs `ReleasedAdaFlashOPD` before training and binds its separate AREPO state contract. The executed objective is the literal released loss in `LOSS_CONTRACT.json`, not ordinary forward KL. The source installer is `online_feedback/adaflash_repo.py` in the execution commit.

`MODEL_PIN.json`, `TRAINING_PIN.json` and `RUNTIME_PIN.json` preserve inherited qualified model/environment/native-runtime identities. The `source_sha256` subsection inside the historical runtime pin is an older foundation snapshot, not a claim that it identifies the current project's source. The execution commit and sealed source identity identify the current code. Similarly, historical free-space fields in these pin files are not current storage measurements.

The serving and training Python environments are separately pinned. No model/data downloads, environment replacements or precision changes were introduced during this campaign. `EXPORT_AUDIT.json` binds retained O58, K+58 and paper-A58; fresh endpoint weight receipts bind all four measured exports. Request texts and token IDs remain in the private sealed plan/raw evidence, outside Git.
