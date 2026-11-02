"""An optional operational barrier before timed final serving, with no model work."""
import hashlib
import json
import os
from pathlib import Path
import time

from online_feedback.storage import write_json


ENVIRONMENT_VARIABLE = "FULL_K_TRANSFER_CLEARANCE"


def marker_path(root, value):
    path = Path(value)
    if (not path.is_absolute() or not path.is_relative_to(root)
            or path == root or path.is_symlink() or path.resolve() != path):
        raise ValueError("Transfer clearance must be a regular path inside this run root")
    return path


def validate_clearance(value, root, plan):
    expected = {
        "schema": "full_k_transfer_clearance_v1",
        "status": "cleared",
        "run_root": str(root),
        "binding": plan["binding"],
        "execution_source": plan["execution_source"],
    }
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Transfer clearance does not match this sealed run")
    for key in ("copy_process_stopped", "transport_processes_stopped", "no_transfers_until_campaign_exit"):
        if value.get(key) is not True:
            raise ValueError("Transfer clearance has no explicit quiescence commitment: " + key)


def wait_for_transfer_clearance(root, plan):
    """Wait for an operator's verified transfer shutdown; the enclosing deadline applies.

    The marker is an operational attestation, not process discovery. Its writer
    must check the actual copy/transport processes and prevent new transfers
    through campaign exit. A missing marker waits; a wrong marker never passes.
    """
    configured = os.environ.get(ENVIRONMENT_VARIABLE)
    if configured is None:
        return {"status": "not_configured", "wait_wall_s": 0.0}
    root = Path(root).resolve()
    receipt_path = root / "campaign_001/TRANSFER_CLEARANCE_WAIT.json"
    started = time.perf_counter()
    receipt = dict(status="waiting", marker_path=configured, binding=plan["binding"],
        execution_source=plan["execution_source"], wait_wall_s=0.0, model_work=0,
        scope="Operational waiting before final backend construction; included in campaign_wall_s")
    write_json(receipt_path, receipt)
    try:
        while True:
            path = marker_path(root, configured)
            if path.exists():
                if not path.is_file():
                    raise ValueError("Transfer clearance marker is not a regular file")
                contents = path.read_bytes()
                clearance = json.loads(contents)
                validate_clearance(clearance, root, plan)
                receipt.update(status="cleared", clearance=clearance,
                    marker_sha256=hashlib.sha256(contents).hexdigest())
                break
            time.sleep(10)
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["wait_wall_s"] = time.perf_counter() - started
        write_json(receipt_path, receipt)
    return receipt
