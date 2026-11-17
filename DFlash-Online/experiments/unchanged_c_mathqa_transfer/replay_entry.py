"""One serial PR51 replay process on the original training-compatible A800."""
import argparse
import os
from pathlib import Path
import time
import traceback

from experiments.persistent_prefix_withholding.qualification_entry import HostMemorySample, install_interrupt_handlers
from online_feedback.storage import write_json
from .runtime import boundary,occupied_card,read


def run(plan,arm,slot="A"):
    if arm not in ("N","S") or slot != "A" or os.environ.get("CUDA_VISIBLE_DEVICES") != plan["devices"]["A"]:
        raise ValueError("Replay must use N/S on the pinned original training-compatible physical GPU A")
    base=Path(plan["root"])/"prerequisite"/f"replay_{arm}5"
    base.mkdir(parents=True,exist_ok=False)
    status=dict(status="running",arm=arm,event=5,binding=plan["binding"],
        execution_commit=plan["execution_commit"],physical_gpu_uuid=plan["devices"]["A"],started_epoch=time.time())
    write_json(base/"STATUS.json",status)
    monitor=HostMemorySample()
    try:
        with occupied_card(plan["root"],f"replay_{arm}5",plan["devices"]["A"]),monitor:
            boundary(plan)
            from .replay import run_replay
            status["result"]=run_replay(plan,arm)
            status["postflight"]=boundary(plan,compiler=True)
            status["status"]="passed"
    except BaseException as error:
        status.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        status.update(wall_s=time.time()-status["started_epoch"],sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes,
                      rss_sample_interval_s=.25,rss_not_continuous_peak=True)
        write_json(base/"STATUS.json",status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--arm",choices=("N","S"),required=True)
    parser.add_argument("--slot",choices=("A",),default="A")
    args=parser.parse_args();install_interrupt_handlers();run(read(args.plan),args.arm,args.slot)


if __name__=="__main__":main()
