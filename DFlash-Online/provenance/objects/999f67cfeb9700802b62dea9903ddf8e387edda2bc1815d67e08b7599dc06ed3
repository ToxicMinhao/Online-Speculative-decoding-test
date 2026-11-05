"""Write one startup provenance receipt; never called from the serving path."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import torch

out = Path(sys.argv[1])
root = Path('/data/sources/minhao/DFlash-Online')
runtime = Path('/data/conda_envs/minhao/adaflash_author/lib/python3.12/site-packages/sglang/srt')
files = ['speculative/dflash_worker.py','speculative/dflash_info.py','layers/logits_processor.py',
         'models/qwen3.py','models/dflash.py','entrypoints/engine.py','server_args.py']
value = {'python':sys.executable,'python_version':platform.python_version(),
         'torch':torch.__version__,'transformers':metadata.version('transformers'),
         'cuda':torch.version.cuda,'free_bytes':shutil.disk_usage(root).free,
         'gpu_name':torch.cuda.get_device_name(0),'gpu_capability':list(torch.cuda.get_device_capability(0)),
         'native_runtime_sha256':{name:hashlib.sha256((runtime/name).read_bytes()).hexdigest() for name in files},
         'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root/'online_feedback').glob('*.py'))}}
out.write_text(json.dumps(value,indent=2)+'\n')
