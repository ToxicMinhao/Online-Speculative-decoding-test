"""Run tiny fixtures with CUDA hidden and confirm no CUDA context was created."""
import json
from pathlib import Path
import sys
import time
import unittest

import torch

root = Path('/data/sources/minhao/runs/dflash_prefix_ranking_implementation_20261025')
assert not torch.cuda.is_initialized()
started = time.perf_counter()
suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
cuda_initialized = torch.cuda.is_initialized()
receipt = {
    'tests_run': result.testsRun, 'failures': len(result.failures),
    'errors': len(result.errors), 'skipped': len(result.skipped),
    'passed': result.wasSuccessful(), 'cuda_initialized': cuda_initialized,
    'seconds': time.perf_counter() - started,
    'python': sys.executable, 'torch': torch.__version__,
    'scope': 'CPU synthetic fixtures only; no released model training or generation',
}
(root / 'CPU_TESTS.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
assert not cuda_initialized
sys.exit(not result.wasSuccessful())
