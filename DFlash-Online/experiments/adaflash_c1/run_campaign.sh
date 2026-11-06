#!/usr/bin/env bash
# Published source, authenticated controls and storage admission precede model work.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference:/data/sources/minhao/DFlash-Online/tests
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

run_root="$1"
run_python=/data/conda_envs/minhao/dflash_specforge_20261022/bin/python
test -f "$run_root/sealed/SEAL.json"
test ! -e "$run_root/START_UTC"
date -u +%FT%TZ > "$run_root/START_UTC"
echo "$$" > "$run_root/SUPERVISOR_PID"
trap 'result=$?; date -u +%FT%TZ > "$run_root/END_UTC"; echo "$result" > "$run_root/EXIT_CODE"' EXIT

CUDA_VISIBLE_DEVICES='' "$run_python" -m unittest discover -s tests -p 'test_adaflash*.py' -v \
  > "$run_root/CPU_TESTS.log" 2>&1
"$run_python" -u -m experiments.adaflash_c1.preflight "$run_root"

# Same48hour operational ceiling as the qualified full-K runner; no efficacy stop.
set +e
timeout --signal=TERM --kill-after=120s 48h \
  "$run_python" -u -m experiments.adaflash_c1.campaign "$run_root"
campaign_exit=$?
echo "$campaign_exit" > "$run_root/CAMPAIGN_EXIT_CODE"
"$run_python" -u -m experiments.adaflash_c1.report "$run_root"
report_exit=$?
"$run_python" -u -m experiments.adaflash_c1.preflight "$run_root" --postflight
postflight_exit=$?
printf 'report=%s\npostflight=%s\n' "$report_exit" "$postflight_exit" > "$run_root/REPORT_EXIT_CODES"
for result in "$campaign_exit" "$report_exit" "$postflight_exit"; do
  if [ "$result" -ne 0 ]; then
    exit "$result"
  fi
done
exit 0
