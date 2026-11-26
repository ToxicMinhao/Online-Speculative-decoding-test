# Point3 commands and execution status

**No A800 model command below has executed.** The current admission receipt is false. These are prospective commands for the unchanged normal-publication route after sufficient capacity and fresh source/input admission; they do not authorize lowering the reserve or deleting inherited artifacts. The source destination and run root have not been staged/created on A800. CPU tests use the new isolated checkout on Fenrir.

## CPU checks used

From `DFlash-Online`, with the existing Fenrir environment:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:training:inference \
/home/minhao2/.venvs/dflash-specforge/bin/python - <<'PY'
import unittest
suite = unittest.defaultTestLoader.discover('experiments/frontier_preservation_pilot', top_level_dir='.')
for module in ('experiments.repair_conditional_kl_pilot.test_evaluation',
               'experiments.repair_conditional_kl_pilot.test_analysis'):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
for pattern in ('test_point3_serving.py', 'test_point1_readout.py',
                'test_point2a_readout.py', 'test_repair_conditional_kl.py'):
    suite.addTests(unittest.TestLoader().discover('tests', pattern=pattern))
result = unittest.TextTestRunner(verbosity=2).run(suite)
import torch
print('CUDA initialized:', torch.cuda.is_initialized())
raise SystemExit(not result.wasSuccessful())
PY
```

The actual final run passed291tests, CUDA uninitialized. Small tensor tests and fake serving backends are CPU implementation evidence; they are not native qualification.

## Required source/data preparation

Before any future native launch, stage the exact clean published checkout at the source path below; make its `SOURCE.json` from tracked files and verify every entry. Record the actual commit, clean status and source/config identities. Re-audit live resources and write `ADMISSION.json` with the full per-filesystem peak and the same limits. Current admission is blocked and must not be edited to claim a pass. `point3_prepare` recomputes the admission fields and refuses a failed plan before model construction.

The normal route needs20GB additional remote peak and0.6GB local allowance. An alternate destination/persistence route is not silently substituted by these commands. All inherited parent/control/feature/witness bytes are authenticated in place; do not copy or regenerate the pool.

## Prospective A800 prepare and supervised run

```bash
set -e
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/conda_envs/minhao/dflash_specforge_20261022
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export PYTHONDONTWRITEBYTECODE=1
cd /data/sources/minhao/point3_cumulative_max_source_20261125_001/DFlash-Online
export PYTHONPATH=.:training:inference
point3_commit=$(python -c 'import json; print(json.load(open("../SOURCE.json"))["commit"])')
CUDA_VISIBLE_DEVICES= python -m experiments.frontier_preservation_pilot.point3_prepare \
  --production-plan /data/sources/minhao/runs/dflash_frontier_preservation_20261125/production_001/sealed/PLAN.json \
  --point2a-root /data/sources/minhao/runs/dflash_point2a_c_reference_20261125/diagnostic_001 \
  --archive-manifest /data/sources/minhao/point2a_c_reference_source_20261125_001/RETAINED_COMPACT_MANIFEST.json \
  --root /data/sources/minhao/runs/dflash_point3_cumulative_max_20261125/final_001 \
  --source-manifest /data/sources/minhao/point3_cumulative_max_source_20261125_001/SOURCE.json \
  --execution-commit "$point3_commit" \
  --admission-path /data/sources/minhao/point3_cumulative_max_source_20261125_001/ADMISSION.json
python -m experiments.frontier_preservation_pilot.point3_launch \
  --plan /data/sources/minhao/runs/dflash_point3_cumulative_max_20261125/final_001/CONTRACT.json
```

Use the established detached server-job wrapper around the final command, retaining PID, logs and exit status. Do not launch when prepare fails. The supervisor owns the clock, serial B-card qualification/events and A-card quiet evaluation; it reserves every Adam call before execution and preserves failures. It does not automatically retry. The handoff permits at most one identified qualification correction retry, but a changed-source retry must retain the original ledger/clock and receive a newly checked source-binding continuation; no full-event restart is supported.

Native qualification must pass before either event; complete paired events precede serving. Future reports must identify their **actual execution commit**, not call this CPU implementation commit a completed execution. No main-run results, generated request rows, checkpoint identities or speedup intervals exist yet.
