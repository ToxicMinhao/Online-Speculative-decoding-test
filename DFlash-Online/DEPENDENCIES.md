# Dependencies and external artifacts

The experiments use one coordinator/training environment and a separate native-serving worker. Installing the latest packages from upstream is not equivalent to the qualified setup.

## Recorded environments

| Item | Training/coordinator | Native-serving worker |
|---|---|---|
| Python |3.12.15|3.12.14|
| PyTorch |2.13.0+cu130|2.9.1+cu130|
| Transformers |5.12.1|4.57.1|
| Main source dependency | SpecForge `1839aae30003e3573d00979eb1edd57347b3578c` | SGLang PR 34171, `b8f7b319c634aac88910aec8ae46e6a1f194ac66` |
| Role | Feedback training, objectives, optimizer/state handling | Greedy DFlash generation, target verification, native feedback features |

These versions come from the latest retained `TRAINING_PIN.json` and `RUNTIME_PIN.json`. The native source hashes and actual package/source manifest provide the remaining runtime identity. A runtime file hash verifies a known file; it does not replace the complete dependency manifest or installation provenance.

The experiment modules expect the project root and the appropriate `training/` and `inference/` paths to be importable, as in the recorded launchers. Machine-specific absolute paths in archived scripts are the paths used for the experiments, not universal defaults. No environment installation or experiment is launched by reading this document.

## Models and tokenizer

| Artifact | Source | Pinned revision |
|---|---|---|
| Target | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Draft | `z-lab/Qwen3-8B-DFlash-b16` | `9b41424b7109f9c5413454f481b09a82b85333f4` |

`MODEL_PIN.json` records the individual model/config/tokenizer file identities. Published source does not include the multi-gigabyte weights. The tokenizer and rendering used by a run must match its pin; recreating a text prompt with a different template is not an exact saved input.

## Data and checkpoint dependencies

Recent one-event screens reuse 128 saved feedback responses and their teacher features, presentation order and sampled-anchor records. The development and additional panels are separate fixed request inventories. Their selection seeds, source revisions, exclusions and previous-exposure disclosures are recorded per run. Dataset answers are not training feedback.

Exact continuation additionally requires the arm's own saved weights, optimizer moments and counters, configuration/event history, training RNG and accumulation state. A model export alone is insufficient. Historical cleanup may have retired some bulky artifacts; consult the run's retention record instead of assuming every old checkpoint remains resumable.

Private prompts, generated response bodies, full token trajectories, teacher tensors, optimizer states and exports stay outside Git. Public reports and compact measurement tables identify what was run without pretending those external artifacts are bundled. Public source plus a report is not a promise that an exact historical rerun can be executed without the retained inputs.

## Which configuration is authoritative?

Use the configuration and arguments sealed for the specific run. `configs/osd.yaml` is a historical recipe and can differ from a later sealed configuration; for example, the recent runs retain a 2,048-token serving output cap with a 2,272-token prepared training-sequence allowance. Do not silently edit either historical record to remove that distinction.

Any future environment upgrade, portability change or model substitution needs its own execution qualification. It should not be attached retrospectively to measurements made under the recorded setup.

The exact seven native serving sources are in [`external_runtime/`](external_runtime/), with the original SGLang fork/commit in [`UPSTREAM.json`](external_runtime/UPSTREAM.json). That directory is an evidence overlay, not a full SGLang distribution.
