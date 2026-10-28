"""CPU-only correctness suite; never initializes CUDA."""
import argparse
import contextlib
import json
from pathlib import Path
import time
import unittest
import torch
parser=argparse.ArgumentParser()
parser.add_argument('--source',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.output)
started=time.perf_counter()
with (root/'CPU_TESTS.log').open('w') as stream, contextlib.redirect_stdout(stream):
    suite=unittest.defaultTestLoader.discover(str(Path(args.source)/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
receipt=dict(status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
             failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
             wall_s=time.perf_counter()-started, cuda_initialized=torch.cuda.is_initialized(),
             python=__import__('sys').executable, source=args.source)
(root/'CPU_TESTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
assert not torch.cuda.is_initialized()
raise SystemExit(not result.wasSuccessful())
