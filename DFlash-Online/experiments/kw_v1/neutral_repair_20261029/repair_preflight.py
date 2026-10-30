"""Read-only admission for the neutral repair; no model or optimizer work."""
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from online_feedback.storage import write_json
from online_feedback.qualify_sglang import check_recipe
from experiments.kw_v1.prepare import identity
from experiments.kw_v1.preflight import hardware_admission, verify_initial_states, verify_qualification_inputs
from experiments.osd_components.prepare import verify_files, verify_native_runtime

root=Path(sys.argv[1]); started=time.perf_counter(); receipt={"status":"running", "model_passes":0,"optimizer_steps":0,"generations":0}
try:
    seal=json.loads((root/"SEAL.json").read_text())
    for path, expected in seal["files"].items():
        if identity(path)!=expected: raise ValueError("Sealed file changed: "+path)
    receipt["files_checked"]=len(seal["files"])
    plan=json.loads((root/"PLAN.json").read_text()); original_root=Path(plan["qualification"]).parent.parent
    originals=json.loads((original_root/"sealed/REMOTE_INPUT_IDENTITIES.json").read_text())
    for path,expected in originals.items():
        if identity(path)!=expected: raise ValueError("Original state changed: "+path)
    receipt["hardware"]=hardware_admission()
    available=int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))*1024
    receipt["storage"]={"free_bytes":shutil.disk_usage(root).free,"additional_allowance_bytes":40*2**30,"reserve_bytes":8*2**30,"host_available_bytes":available,"host_required_bytes":80*2**30}
    if receipt["storage"]["free_bytes"]<48*2**30 or available<80*2**30: raise ValueError("Qualification resource budget not admitted")
    pins=json.loads((original_root/"sealed/MODEL_PIN.json").read_text())
    for name in ("target","draft"): verify_files(plan[name],pins[name]["files"],exact_files=True)
    receipt["recipe"]=check_recipe(plan["config"])
    receipt["native_runtime"]=verify_native_runtime(plan["worker_python"],json.loads((original_root/"sealed/RUNTIME_PIN.json").read_text()))
    import platform, transformers
    actual={"python":platform.python_version(),"torch":torch.__version__,"transformers":transformers.__version__}
    if actual!=json.loads((original_root/"sealed/TRAINING_PIN.json").read_text())["environment"]: raise ValueError("Training runtime changed")
    receipt["training_runtime"]=actual
    receipt["states"]=verify_initial_states(plan,originals)
    receipt["features"]=verify_qualification_inputs(original_root,plan)
    if torch.cuda.is_initialized(): raise ValueError("Read-only preflight initialized CUDA")
    receipt["status"]="passed"
except BaseException as error:
    receipt.update(status="failed",error=repr(error)); raise
finally:
    receipt["wall_s"]=time.perf_counter()-started
    write_json(root/"PREFLIGHT.json",receipt)
