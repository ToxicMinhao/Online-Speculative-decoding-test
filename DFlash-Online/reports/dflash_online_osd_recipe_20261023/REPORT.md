# DFlash-Online selected OSD recipe — implementation report

Implemented and installed on A800, October 23, 2026. No scientific experiment,
full-model GPU qualification, or paused campaign was started. The project stage
and research direction are unchanged.

## Selected recipe

A800 configuration: `/data/sources/minhao/DFlash-Online/configs/osd.yaml`.
Use with `SpecForgeEventTrainer(..., objective="osd", reduction="response_mean")`
and a fresh optimizer/output directory. The existing qualification command accepts
this explicit configuration; its historical default objective remains upstream CE.

| Setting | Current selected OSD recipe |
|---|---|
| Learning rate | Constant 3e-4, no warmup or scheduler |
| Optimizer | Direct torch.optim.AdamW on BF16 draft parameters and moments; no FP32 masters or CPU offload |
| Weight decay | 0.01; betas (0.9, 0.999), epsilon 1e-8 unchanged |
| Response order | Fresh Torch RNG permutation each pass; actual presentation order recorded |
| Anchors | Up to 512 sampled per response, limited by existing eligible positions |
| Probability/KL arithmetic | BF16 softmax, log-softmax and KL rows; existing FP32 position weights, weighted reduction and denominator |
| Draft training attention | Existing SpecForge Flex attention backend |

Two passes and accumulation two remain: eight responses give sixteen response
presentations and eight optimizer steps. The configuration retains the current
screen's eight-step horizon. This does not authorize or schedule that screen.

## Preserved boundaries

Anchor eligibility still requires the anchor and next token to be supervised.
Sequences retain exact saved token IDs, masks and lengths, with no padding,
retokenization or silent truncation. Earlier context receives no new loss.
Serving remains the existing official HF/SDPA implementation. No edit was made
to upstream `training/`, upstream `inference/`, feedback capture/preparation,
or Fenrir `/home/minhao2/DFlash-Research`.

Historical configs and experiment drivers are retained unchanged. Their default
FP32-master optimizer, FP32 KL and fixed order still reproduce their prior recipe.
The historical `experiments/osd_followup` driver has old matched-exposure/RNG gates;
it is not a launcher for this revised recipe. Do not silently reuse it as such.
Old states cannot resume into the new recipe: config and algorithm contracts
reject incompatible optimizer/numerical settings before applying state.

## Code and verification

`recipe.py` parses only online options before upstream strict config validation;
`optimizer.py` adapts native BF16 AdamW to TrainerCore; `training.py` selects the
optimizer and shuffles each pass; `objective.py` selects probability/KL precision.
Only nine listed source/config/test/documentation files were installed. See
[exact file list](INSTALL_FILES.json) and [installation receipt](INSTALL_RECEIPT.json).
All 42 remaining files in the prior source archive were compared byte-for-byte
and preserved. Separate read-only review found no blocking integration issue.

**177 CPU tests pass on both Fenrir and A800.** Tests include direct PyTorch AdamW
update/moment equivalence, rejection of nonfinite gradients before mutation,
BF16 KL/gradient equivalence with FP32 weighting, unchanged masks and eligibility,
512-anchor cap, sixteen presentations/eight constant-LR steps, and exact resumed
response order, anchors, weights, optimizer state and RNG. Historical tests pass.
Logs: [Fenrir](full_cpu.log), [A800](cpu_tests.log).

The tiny CPU integration fixture uses eager attention deliberately. Config wiring
to Flex is checked; full-model GPU compilation, peak memory and serving output
preservation with this new recipe remain unqualified. CPU checks establish no
learning benefit. No efficacy result is being replaced or reinterpreted.

## Storage and recovery

Source-only budget: 256 MiB plus an 8 GiB reserve. A800 free space was
28,116,791,296 bytes before installation and 28,115,767,296 bytes after checks;
about 26.18 GiB remains. Both GPUs were idle before these CPU-only checks.
No deletion was necessary or performed. The complete prior online source archive
is retained locally at `before/a800_source.tar` and on A800 at
`/data/sources/minhao/runs/dflash_online_osd_recipe_20261023/implementation_001/a800_source.tar`.
Models, results, checkpoints and environments were not changed.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_recipe_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
