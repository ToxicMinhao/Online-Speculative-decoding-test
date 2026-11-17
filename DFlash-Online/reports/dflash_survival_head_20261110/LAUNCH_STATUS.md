# Post-PR36 queue and sealed execution

Execution code: `7f89719956f0256c0916a583d7ff615954cc66f9` ([PR37](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/37)). This documentation publication does not change the staged execution code.

At 23:32 UTC November 10, the detached CPU queue is alive and waiting for PR36, with zero pilot execution attempts. PR36 is training MathQA event 5 RANDOM on physical GPU 1; its GSM8K phase is complete. The survival pilot has not yet performed native qualification or training.

The A800 source copy verifies 3,737 files and one symlink against the clean published commit. The actual-path plan is sealed with binding `cd68ee3a4c6baf2dacd044e4fee3838c41abf9a1c820b7566166ce0ef3961a1c`; input-only 128-request panel binding is `5668f61d01166ffd310ce98d5eb449d13b46258f28ba2ad7aefa5f5661f2cc73`.

All 151 CPU tests pass. Separate same-team source reviews cover feedback reconstruction, root orchestration and the final queue; prior authorship/shared context are disclosed. Native correctness, memory fit and efficacy remain unestablished.

The queue starts model work only after PR36 exits successfully and fresh source, original-state, runtime, physical-device, storage and quiet-host checks pass. Fixed peak budget 59,916,015,137 bytes includes an 8 GiB reserve. Qualification and the prescribed three-arm event follow once each, without retries or tuning. Endpoint timing checks both cards and transfer activity. Expected work after the window opens remains 1.5–2.5 hours, including 45–60 minutes of quiet serving.

A slow full-source transfer was replaced with verified on-host-base plus a small Git delta. A transfer checker initially rejected Git's root directory entry; its directory-only handling was corrected. Neither operation performed model work or changed scientific code. The abandoned source-only staging directory was retired after verified replacement; exact 2,215-file retirement evidence remains private on both execution/preparation roots. No scientific artifact was deleted.
