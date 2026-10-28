#!/usr/bin/env bash
set -u
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1
ranking_root=/data/sources/minhao/runs/dflash_prefix_ranking_one_event_20261025
printf '%s\n' "$$" > "$ranking_root/SUPERVISOR.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$ranking_root/STARTED.utc"
bash "$ranking_root/run_campaign.sh"
ranking_exit=$?
printf '%s\n' "$ranking_exit" > "$ranking_root/EXIT_CODE"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$ranking_root/FINISHED.utc"
exit "$ranking_exit"
