# Native serving and checkpoint Python sources

The seven `sglang/srt` files are exact installed sources used by the native worker, archived with their Apache-2.0 license. They are an overlay/evidence subset, **not a complete SGLang installation**. The base is SGLang PR 34171, commit `b8f7b319c634aac88910aec8ae46e6a1f194ac66`; the measured environment is separately pinned in each run. Use the original dependency source and verify the retained per-file digests, rather than installing current SGLang.

`model_code/` contains only Python source distributed with the pinned Qwen3-8B-DFlash-b16 checkpoint, not model tensors. Its pinned revision is `9b41424b7109f9c5413454f481b09a82b85333f4`. These files retain their original notices; the DFlash MIT license accompanies them.

Training and serving used separate Python environments. See the root dependency documentation and the per-run runtime manifests. This publication did not reinstall dependencies or repeat a GPU experiment.
