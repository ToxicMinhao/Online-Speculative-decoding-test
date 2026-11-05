# Source publication, October 28, 2026

The experimental source is published at commit [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a). This is an archive created after execution, not an original execution-time revision. Each report identifies the historical variants and hash contracts used for its study.

## What is preserved

- Full A800 DFlash-Online source, including upstream DFlash/SpecForge, exact compatibility patches, shared online-feedback code and bounded experiment runners.
- 855 captured non-documentation entries verified byte-for-byte; the three edited root documents have exact originals in `provenance/live_capture/`. No scientific Python file was changed.
- 969 archived source variants, 7,482 verified source-pin occurrences, 182 readable launch/analysis scripts and 85 configuration/pin copies.
- 177 report copies from 37 groups, with 38 study/preparation groups inventoried. Maintained local reports received provenance appendices; 37 immutable raw copies received sidecars without changing their bytes. Original report hashes remain in the index.

## Checks

Nineteen focused CPU tests passed from the published layout. Eight archive-helper tests and two provenance-writer tests passed. Archive verification checks every stored object, pin and readable copy. The first CPU invocation used an older environment missing `pydantic`; it did not run two modules. The existing qualified SpecForge environment then passed all 19 tests without a code change. Separate source/structure and privacy reviews were performed by same-team agents; this is not an external peer review.

No model training, generation, checkpoint modification or new GPU qualification was performed. This validates publication fidelity and CPU behavior, not a fresh reproduction of experiment results.

## Explicit limits

Nine old source pins remain unavailable: seven initial foundation revisions and two pre-reseal test revisions. The manifests identify them rather than substituting present-day code. Survey-only OSD reference files without verified redistribution terms remain upstream links/hash records; the independent DFlash-Online implementations are published. Private prompts, responses, token trajectories, features, model exports and optimizer states remain outside Git.

The A800 working path is unchanged. `GITHUB_SOURCE.json` binds it to this publication; the original directory was not a Git checkout, and publication does not rewrite that history. Future runs must record their actual source revision and any local changes before execution.
