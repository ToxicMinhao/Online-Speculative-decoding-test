"""CPU-only tests on the installed DFlash-Online source; no scientific run."""
import json
import time
import unittest
from pathlib import Path
import torch

root = Path('/data/sources/minhao/runs/dflash_reference_margin_implementation_20261026')
assert not torch.cuda.is_initialized()
started = time.perf_counter()
suite = unittest.defaultTestLoader.discover('/data/sources/minhao/DFlash-Online/tests')
result = unittest.TextTestRunner(verbosity=2).run(suite)
summary = {'tests': result.testsRun, 'failures': len(result.failures),
           'errors': len(result.errors), 'skips': len(result.skipped),
           'seconds': time.perf_counter() - started,
           'cuda_initialized': torch.cuda.is_initialized(), 'torch': torch.__version__,
           'device': 'CPU', 'success': result.wasSuccessful()}
(root / 'CPU_SUITE_A800.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
assert not torch.cuda.is_initialized()
raise SystemExit(0 if result.wasSuccessful() else 1)
