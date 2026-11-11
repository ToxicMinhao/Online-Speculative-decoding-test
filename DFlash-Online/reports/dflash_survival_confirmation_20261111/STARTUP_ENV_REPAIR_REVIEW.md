## Material Passport

2026-11-11, focused startup-environment repair review. Inputs: retained preparation/STARTUP_FAILURE.txt, original survival_head/run_all.sh, repaired confirmation launcher and unchanged SGLang backend. Scope: local CPU/source validation only; same-team review and test authorship overlap are disclosed. No remote commands, model imports, model execution or production edits by this reviewer. Stage 1 / project None unchanged.

The retained execution001 failure is concrete: SGLang CUDA-graph startup entered the fused RoPE JIT link, invoked bare `c++`, and exited127 because that executable was unavailable. This occurred before reference generation attempts according to the executor's retained counters. Startup/model-loading and graph activity did occur; zero generated responses does not mean zero startup GPU work. Preserve those failure records separately from the repaired execution.

The root's launcher repair restores the completed pilot's exact CC/CXX paths, controller CONDA_PREFIX/PATH prefix, four-thread OMP/MKL/OpenBLAS settings, offline switches, tokenizer-parallelism setting and bytecode suppression. The added executable checks reject either missing/nonexecutable compiler before any Python process starts. The native backend retains these compiler settings and derives CUDAHOSTCXX/NVCC_CCBIN from CXX; no kernel, model, algorithm, sample or scientific setting is changed.

No remaining consequential launcher-environment discrepancy was found. Confirmation's code-only PYTHONPATH is sufficient for this serving-only entry point; both paths explicitly set the native worker's PYTHONPATH to the same code root. The backend applies its pinned worker Conda environment/library path and the sealed serving_gpu itself. Therefore the original pilot's training-parent CUDA_VISIBLE_DEVICES=1 versus confirmation-parent0 does not change the physical serving device; existing UUID and quiet-host checks still apply. Both launchers source the same required A800 setup file. Live compiler/runtime availability is the executor's admission responsibility, not established by the local fixtures.

Added tests/test_survival_confirmation_launch_env.py. All four focused tests passed in0.056s using the existing local dflash-specforge Python with CUDA_VISIBLE_DEVICES empty: Bash syntax; actual shell evaluation of common startup environment against the pilot; full repaired launcher execution with harmless Python recorders observing runtime/report environments; and fail-before-Python admission for each nonexecutable compiler. The tests replace only remote paths and setup with temporary local fixtures and do not import Torch/SGLang or execute the experiment. git diff --check passed. These four tests supplement the existing42; the root owns the final combined rerun/publication and fresh execution002 seal.

The root reports that an earlier combined CPU run overlapped construction of the new fixture and caught its temporary missing-parent-directory error; CPU_REPAIR_TESTS.log preserves that failed check. The fixture was corrected before the focused four-test run above. This was a CPU test-fixture failure with no model work. The root is completing the combined46-test rerun.

The repair is appropriate for the observed missing-compiler failure. These CPU checks do not qualify native RoPE linking or promise subsequent startup success. The previously authorized algorithm/data/settings and stopping rules remain unchanged; this review does not authorize an automatic retry mechanism.

Reviewed repair is relative to cf03c95e8f7e268537d0e2eb81a622419ba1b4ca. The root handles publication; the fingerprints below bind the tested bytes independently of publication timing.

| File | SHA-256 |
| --- | --- |
| `experiments/survival_confirmation/run_all.sh` | `b984c0bd7851ac3f29bd10b8790fd151fa57fd49d260fcdf87a00ae9e5c4dd49` |
| `tests/test_survival_confirmation_launch_env.py` | `1160b51044f72d1d77b7bdc00bfd0a7b64a1c3eefaf2c7be88e941a2cc7e4e8e` |


Executor completion addendum: actual successful execution commit `e0129ee217f3f5b4bf6bb62bc44652e7d47b737d`. All 4,608 timed generations and native postflight passed; independent raw-row arithmetic now passed all 75 numerical checks and all 12 bootstrap intervals, with exact 4,608-row CSV coverage. Earlier statements of pending native or arithmetic work describe that review’s original time and are superseded by the final receipts. This addendum does not make the earlier scoped review a retrospective independent GPU audit.
