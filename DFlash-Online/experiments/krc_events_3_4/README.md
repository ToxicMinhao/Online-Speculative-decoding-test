# KRC continuation through event 4

This bounded runner resumes full-frequency KRC2's complete corrected state and
runs events 3 and 4 with the existing K objective and reference correction.
It adds orchestration only; inherited scientific implementations are unchanged.

- Start: cumulative256 Adam steps /512 ordinary presentations, empty accumulation.
- Event3: retain complete KRC3 state and export at384/768.
- Event4: retain complete KRC4 state and export at512/1024.
- Preserve the7424-step horizon, optimizer/RNG continuity, retained O-generated
  pools and naturally sampled presentation/anchor bindings. No new feedback.
- New work:512 ordinary presentations,256 Adam steps,1536 core response forwards,
  at most1024 backwards and256 direct corrections. Optional probes are disabled.
- Qualified disposable startup:2 forward/backwards,zero Adam, separately counted.

After closing training, compare retained O4/K4 with KRC4 on the unchanged
64-request development and64-request seed20261028 additional panels. Three
rotating repeats give1152 generations. The initial8 development requests form
an included72-execution gate. All outputs and terminations must match retained
canonical records; development O4/K4 per-request counts must also reproduce.
There is no historical O4/K4 count gate on the additional panel.

Both complete states are retained as hard links to the trainer's atomically
replaced latest checkpoint. Full-state/export and live-boundary checks require
saving KRC3 to leave the trajectory unchanged; final identities check both
states survive event4. Recovery paths in each external RESULT.json are immutable.

The runtime entry point is:

```bash
python -m experiments.krc_events_3_4.campaign /path/to/sealed/run
```

Use the committed preparation/preflight/launch tools under
`provenance/run_tools/dflash_krc_events_3_4_20261028/` to bind actual artifacts,
source revision, storage admission and runtime. Private response tokens, feature
pools and recovery states remain outside Git. The command performs the authorized
experiment and must not be used with substituted fixtures or unsealed inputs.

Report panels separately. O is the primary OSD baseline; K is the correction
ablation. Count curves use events1/2/4, but historical versus fresh timings do not
measure learning progress. KRC4 versus K4 measures the complete correction history,
not an isolated effect of events3/4. No tuning or continuation follows this run.
