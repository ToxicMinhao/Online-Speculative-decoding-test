#!/usr/bin/env bash
set -u
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1
one_event_root=/data/sources/minhao/runs/dflash_osd_one_event_20261025
printf '%s\n' "$$" > "$one_event_root/SUPERVISOR.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$one_event_root/STARTED.utc"
bash "$one_event_root/run_campaign.sh"
one_event_exit=$?
printf '%s\n' "$one_event_exit" > "$one_event_root/EXIT_CODE"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$one_event_root/FINISHED.utc"
exit "$one_event_exit"
