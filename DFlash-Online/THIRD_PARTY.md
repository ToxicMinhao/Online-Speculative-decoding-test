# Third-party source and attribution

DFlash-Online builds on DFlash and SpecForge. Our online-feedback and experiment modules reuse these implementations and add the documented research objectives, controls and accounting. Routine reuse does not imply a new inference architecture.

| Component | Upstream | Pinned revision | License |
|---|---|---|---|
| DFlash inference | https://github.com/z-lab/dflash | `07ebd93db9f472af339b644bb70221ad8428328a` | MIT, Copyright 2026 Z Lab; retain `inference/LICENSE` |
| SpecForge training | https://github.com/sgl-project/SpecForge | `1839aae30003e3573d00979eb1edd57347b3578c` | MIT, Copyright 2025 sgl-project; retain `training/LICENSE` |

The pinned source tree and local compatibility patches belong in the publication provenance. Inference changes include the feedback trace hook and cache compatibility handling used by this project. The SpecForge cache-directory change belongs to the training-source history. Actual file identities, not a prose list, define each executed version.

The recent native-serving campaigns also use an external SGLang/RedKnot runtime in a separate environment. That runtime is not the same thing as the bundled `inference/` package. Its exact source identity, measured file hashes and any local changes must be taken from the published runtime provenance and each report's `RUNTIME_PIN.json`. Do not substitute a newer runtime because it shares the SGLang name.

AdaFlash's author-path OSD recipe informed the earlier baseline implementation and protocol work. The current `online_feedback/` code is the project's implementation; the presence of an environment named `adaflash_author` does not mean the campaign imports AdaFlash training code. Reports identify the actual objectives and paths used.

The released target and draft checkpoints are external dependencies, with their own model terms and identities. They are not covered by a source-code license for this repository. The publication does not include their weight tensors or claim ownership of them.

Preserve upstream LICENSE/NOTICE files and any per-file notices when copying source. Nested upstream `.git` directories and credentials are not part of the vendored source distribution. The root repository's existing license decisions remain unchanged by this documentation.

SGLang serving uses PR 34171 at `b8f7b319c634aac88910aec8ae46e6a1f194ac66` ([source fork](https://github.com/flowermouse/sglang/tree/b8f7b319c634aac88910aec8ae46e6a1f194ac66)); its seven archived files retain the Apache-2.0 license in `external_runtime/sglang/LICENSE`. Source/license mappings are in `provenance/licenses/`. The checkpoint Python sources carry the DFlash MIT terms recorded in the model card.

Survey-only OSD author sources without verified redistribution terms are not copied into the public archive. Their pinned upstream links and file hashes remain in `provenance/PUBLICATION_EXCLUSIONS.json`; local originals remain retained. This exclusion does not remove the project's independently implemented experimental source.
