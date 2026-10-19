"""Bounded read-only worker for five retained-correction source-response cases.

The ``train`` allocation is used only for frozen-feature reconstruction. No
trainer, optimizer, bank update, sampler or response generation is constructed
there. Serving admits exactly the fixed F/singleton pair for each declared case.
Durable claims are written before work, so failures cannot silently cause retries.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.dflash_correction_bank import native
from experiments.dflash_correction_bank.artifacts import atomic_json, binding, checked, load_snapshot

base = native.base
require, digest, write_json = native.require, native.digest, native.write_json
FIXED_CASES = (("wildchat_00", 1, 0), ("wildchat_01", 2, 1),
               ("wildchat_02", 6, 3), ("wildchat_19", 15, 2),
               ("personamem_persona42", 3, 2))


def approved_cases(config):
    """Validate immutable input/state bindings before admitting any model work."""
    cases = config.get("retained_cases", [])
    require(len(cases) == len(FIXED_CASES), "Exactly the five authorized cases are required")
    expected = {f"{stream}:t{turn}:m{module}": (stream, turn, module)
                for stream, turn, module in FIXED_CASES}
    require({case["case_id"] for case in cases} == set(expected), "Fixed replay cases changed")
    for case in cases:
        stream, turn, module = expected[case["case_id"]]
        require(case["turn"] == turn and case["module"] == module, "Fixed source turn/module changed")
        require(case.get("stream", stream) == stream, "Fixed source stream changed")
        require(case["input_ids_sha256"] == digest(case["input_ids"]) and
                case["output_ids_sha256"] == digest(case["output_ids"]), "Retained token IDs changed")
        base.check_input(case["input_ids"])
        require(0 < len(case["output_ids"]) <= 1024 and
                all(type(token) is int and token >= 0 for token in case["output_ids"]),
                "Invalid retained canonical output IDs")
        for key in ("after_snapshot", "projections", "feedback", "source_native_config"):
            checked(case[key])
        source = json.loads(checked(case["source_native_config"]).read_text())
        require(source["episode_id"] == stream and not source.get("bank_qualification", False),
                "Use main-campaign snapshots, not qualification snapshots")
        for key in ("released", "target", "runtime_source_sha256"):
            require(source[key] == config[key], "Replay model/runtime differs from retained case: " + key)
        from experiments.dflash_correction_bank.retained_hooks import selected_matrix
        selected_matrix(case)
    require(base.OUTPUT_CAP == 1024 and base.BLOCK_SIZE == 10,
            "Retained serving must preserve block10 and output cap1024")
    return {case["case_id"]: case for case in cases}


def _sd_scheduler(*args, **kwargs):
    config = base._worker_config()
    entry = base._serving_entry(config)
    from experiments.dflash_correction_bank import retained_hooks
    receipt = retained_hooks.install(entry, config)
    write_json(Path(config["worker_out"]) / "RETAINED_NATIVE_HOOKS.json", receipt)
    return native._original_sd_scheduler(*args, **kwargs)


class RetainedServingService(native.ServingService):
    def __init__(self, config):
        self.cases = approved_cases(config)
        self.claims = Path(config["worker_out"]) / "generation_claims"
        self.claims.mkdir(exist_ok=False)
        self.used = set()
        # Install only the fixed correction hook, never the causal read/router.
        base._sd_scheduler = _sd_scheduler
        super().__init__(config)

    def bank_activate(self, body):
        state = body["state"]
        matches = [case for case in self.cases.values()
                   if case["after_snapshot"]["sha256"] == state["state_sha256"]]
        require(len(matches) == 1, "Only an authorized retained singleton may be activated")
        case = matches[0]
        require(state.get("snapshot") == case["after_snapshot"] and
                body["projections"] == case["projections"], "Retained activation bindings changed")
        return super().bank_activate(body)

    def generate(self, body):
        case_id = body["case_id"]
        require(case_id in self.cases, "Unapproved replay case")
        case, arm = self.cases[case_id], body["arm"]
        require(arm in ("F", "BANK"), "Only frozen and retained singleton are authorized")
        require(body["capture"] is False and body["canonical"] is False and
                body["checkpoint_id"] == "F", "Read-only replay must use released full weights without capture")
        require(body["turn"] == case["turn"] and body["input_ids"] == case["input_ids"],
                "Native replay must use the exact source-response input")
        require(body["projections"] == case["projections"], "Retained projection artifact changed")
        state = body["state"]
        require(state["arm"] == arm, "Replay state/method differs")
        if arm == "BANK":
            require(state["snapshot"] == case["after_snapshot"] and
                    state["state_sha256"] == case["after_snapshot"]["sha256"],
                    "Singleton must come from the exact after-turn snapshot")
        else:
            require(state["state_sha256"] == self.config["released"]["checkpoint_identity_sha256"],
                    "Frozen reference must be the released checkpoint")
        key = (case_id, arm)
        require(key not in self.used and len(self.used) < 10, "A retained replay may run only once")
        claim = atomic_json(self.claims / (digest(list(key)) + ".json"), dict(
            case_id=case_id, arm=arm, input_ids_sha256=case["input_ids_sha256"],
            canonical_output_ids_sha256=case["output_ids_sha256"],
            state_sha256=state["state_sha256"], capture=False, learning=False,
            claimed_before_native_execution=True))
        self.used.add(key)
        result = super().generate(body)
        result.update(case_id=case_id, retained_arm=arm, generation_claim=claim,
                      routing_bypassed=True,
                      timing_scope="Routing-bypassed source-response diagnostic; not deployment speedup")
        atomic_json(self.claims / (digest(list(key)) + ".result.json"), result)
        return result


class RetainedPrecisionService:
    """Minimal frozen author runtime; deliberately not a TrainerService subclass."""
    def __init__(self, config):
        self.config, self.out = config, Path(config["worker_out"])
        self.cases = approved_cases(config)
        self.claims = self.out / "reconstruction_claims"
        self.claims.mkdir(exist_ok=False)
        self.used = set()
        directory = Path(config["workspace_root"]) / "research/target_variant_specialization_001/training_tools"
        sys.path.insert(0, str(directory))
        import native as qualified_native
        require(Path(qualified_native.__file__).resolve() == directory / "native.py", "Frozen runtime import collision")
        self.native = qualified_native
        self.runtime = qualified_native.load_runtime(17, base.OUTPUT_CAP)
        require(Path(qualified_native.CORE.AUTHOR).resolve() == Path(config["author_root"]).resolve(),
                "Runtime imported a different author tree")
        args = copy.copy(self.runtime.default_args)
        require(config["released"]["path"] == str(qualified_native.CORE.DRAFT) and
                config["target"]["model_path"] == str(qualified_native.CORE.TARGET), "Frozen model pair changed")
        args.target_model_path = config["target"]["model_path"]
        args.initial_draft_path = args.draft_config_path = config["released"]["path"]
        args.max_length = args.sglang_context_length = base.CONTEXT_LIMIT
        require(args.target_model_backend == "sglang" and args.sglang_attention_backend == "fa3",
                "Keep the qualified SGLang teacher backend")
        self.teacher = self.runtime.setup.load_target_model(args)
        import torch
        self.teacher_model = self.teacher.model_runner.model.eval().requires_grad_(False)
        self.components = self.runtime.setup.load_target_embeddings(args).eval().requires_grad_(False)
        self.draft, self.model_config, _, mask = self.runtime.setup.load_draft_stack(args, self.teacher)
        require(mask == 128002 and self.draft.block_size == 10 and
                self.draft.target_layer_ids == [1, 8, 15, 22, 29], "Released DFlash architecture changed")
        parameters = list(self.draft.parameters())
        require(len(parameters) == 58 and sum(p.numel() for p in parameters) == 1048626432 and
                all(p.dtype == torch.bfloat16 for p in parameters), "Released BF16 parameter inventory differs")
        require(float(getattr(self.model_config, "attention_dropout", 0.)) == 0. and
                not any(getattr(module, "gradient_checkpointing", False) for module in self.draft.modules()),
                "Frozen backbone checkpointing/dropout differs")
        qualified_native.CORE.verify_checkpoint(self.draft, config["released"]["path"])
        self.draft.eval().requires_grad_(False)
        self.versions = self._versions()
        self._check_frozen()
        write_json(self.out / "FROZEN_PRECISION_RUNTIME.json", dict(
            status="passed", args=vars(args), training_enabled=False, optimizer_constructed=False,
            duplicate_bank_draft=False, reconstruction_limit=5,
            teacher_context_limit=args.sglang_context_length, serving_input_limit=base.INPUT_LIMIT,
            reconstruction="Exact saved response tokens and retained anchors/weights; no sampling"))

    def _versions(self):
        return tuple((owner, name, id(parameter), parameter._version)
                     for owner, module in (("teacher", self.teacher_model), ("head_embedding", self.components),
                                           ("draft", self.draft))
                     for name, parameter in module.named_parameters())

    def _check_frozen(self):
        require(self._versions() == self.versions and
                all(not module.training and not parameter.requires_grad and parameter.grad is None
                    for module in (self.teacher_model, self.components, self.draft)
                    for parameter in module.parameters()), "Frozen reconstruction weights changed")

    def _synchronize(self):
        import torch
        if next(self.draft.parameters()).device.type == "cuda":
            torch.cuda.synchronize()

    def teacher_features(self, ids):
        import torch
        with torch.no_grad():
            result = self.teacher.generate_dflash_data(ids, torch.ones_like(ids),
                                                       torch.zeros_like(ids, dtype=torch.float32))
        final = result.teacher_hidden_states
        require(final is not None and final.shape == (*ids.shape, self.components.lm_head.in_features),
                "Native normalized final-layer teacher features missing")
        require(result.hidden_states.shape[:2] == ids.shape and result.hidden_states.shape[-1] == 5*final.shape[-1],
                "Native selected layer features missing or shortened")
        return result.hidden_states.detach(), final.detach()

    def precision(self, body):
        case_id = body["case_id"]
        require(case_id in self.cases and case_id not in self.used,
                "Precision reconstruction is restricted to once per authorized case")
        case = self.cases[case_id]
        claim = atomic_json(self.claims / (digest(case_id) + ".json"), dict(
            case_id=case_id, feedback=case["feedback"], projections=case["projections"],
            after_snapshot=case["after_snapshot"], matrix_sha256=case["matrix_sha256"],
            claimed_before_reconstruction=True, no_retry=True, no_resampling=True))
        self.used.add(case_id)
        started = time.perf_counter()
        try:
            self._check_frozen()
            from experiments.dflash_correction_bank.projections import FixedProjections
            from experiments.dflash_correction_bank.retained_hooks import selected_matrix
            from experiments.dflash_correction_bank.retained_precision import reconstruct_retained_precision
            projections = FixedProjections.load(checked(case["projections"]))
            feedback = load_snapshot(case["feedback"])
            response = feedback["response"]
            require(response["input_ids"] == case["input_ids"] and response["output_ids"] == case["output_ids"]
                    and response["turn"] == case["turn"], "Retained feedback belongs to another source response")
            require(response["target_identity_sha256"] == self.config["target"]["identity_sha256"],
                    "Retained feedback target identity differs")
            E = self.components.lm_head.weight
            require(projections.U.shape == (E.shape[1], 64) and projections.P.shape == (E.shape[0], 64)
                    and projections.V_in.shape == (E.shape[1], 64) and projections.seed == 17,
                    "Retained projection geometry differs")
            evidence = reconstruct_retained_precision(self.draft, self.components.embed_tokens,
                self.teacher_features, E, projections, feedback, selected_matrix(case),
                synchronize=self._synchronize, mask_token_id=128002, attention_backend="flex_attention")
            self._check_frozen()
        except Exception as error:
            evidence = dict(status="unavailable", reason=repr(error), traceback=traceback.format_exc(),
                precision=None, no_retry=True, no_resampling=True,
                interpretation="Unavailable precision is not evidence of useful or ineffective native drafting.")
        evidence.update(case_id=case_id, reconstruction_claim=claim, total_analysis_s=time.perf_counter()-started)
        reference = atomic_json(self.claims / (digest(case_id) + ".result.json"), evidence)
        return dict(status="passed", case_id=case_id, evidence=evidence, evidence_binding=reference)

    def close(self):
        self.runtime.destroy_distributed()


def worker_main():
    """Same authenticated local RPC, with a strictly smaller action surface."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    config = base._worker_config()
    native.check_sources(config)
    numerical = native.check_numerical_environment()
    write_json(Path(config["worker_out"]) / "BANK_NUMERICAL_ENVIRONMENT.json", numerical)
    base.check_dependencies(config["workspace_root"])
    allocation = os.environ.pop("CUDA_VISIBLE_DEVICES")
    try:
        base.check_environment(config)
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = allocation
    require(config["kind"] in ("sd", "train"), "Only serving and frozen precision worker allocations exist")
    service = RetainedPrecisionService(config) if config["kind"] == "train" else RetainedServingService(config)
    poisoned = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                require(self.headers.get("X-Two-Update-Token") == config["rpc_token"], "Foreign worker client")
                length = int(self.headers.get("Content-Length", "0"))
                require(0 < length <= 64*2**20, "Invalid RPC payload")
                body = json.loads(self.rfile.read(length))
                action = self.path.lstrip("/")
                require(not poisoned, "Worker poisoned after prior failure")
                if action == "ready":
                    value = dict(status="passed", pid=os.getpid())
                else:
                    allowed = {"precision"} if config["kind"] == "train" else {"generate", "flush", "bank_activate"}
                    require(action in allowed, "Worker action is outside the retained diagnostic")
                    value = getattr(service, action)(body)
                status = 200
            except Exception as error:
                poisoned.append(repr(error))
                value = dict(status="failed", error=repr(error), traceback=traceback.format_exc())
                write_json(Path(config["worker_out"]) / f"FAILURE_{len(poisoned)}.json", value)
                status = 500
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", config["port"]), Handler)

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    require(sys.argv[1:] == ["--worker"], "Use the separate retained replay controller")
    worker_main()
