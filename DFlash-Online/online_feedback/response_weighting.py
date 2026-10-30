"""KW-v1 bookkeeping around the unchanged, direct BF16 AdamW update.

No model calls, RNG draws, or live optimizer updates occur here. The eligibility
is a finite secant of actual quantized optimizer arithmetic, not an exact
derivative through BF16 or future Adam history.
"""

import copy
import math
import time
from dataclasses import dataclass

import torch


MAX_CHUNK_ELEMENTS = 1_048_576
KW_CONTRACT = {
    "algorithm": "kw_v1", "version": 1, "features": 5,
    "allocation_cap": 0.25, "secant_half_width": 1 / 16,
    "rms_decay": 0.95, "credit_epsilon": 1e-12,
    "credit_clip": 1.0, "controller_learning_rate": 0.05,
    "controller_norm_cap": 2.0, "dot_chunk_elements": MAX_CHUNK_ELEMENTS,
    "feature_dtype": "torch.float32", "controller_dtype": "torch.float64",
    "gradient_dtype": "torch.bfloat16", "eligibility_dtype": "torch.float32",
    "recovery_boundary": "complete_event",
}


def _finite(tensor, label):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Nonfinite {label}")


@torch.no_grad()
def response_features(loss, sum_metrics, response_length):
    """Read one StepOutput's detached observations, before metric merging.

Length is the full response including EOS. Empty blocks are checked, but are
absent from both the first-error numerator and its nonempty-block denominator.
"""
    keys = ["available_position_weight", "selected_position_weight",
            "no_error_blocks"]
    keys += [f"first_error_position_{position}" for position in range(1, 16)]
    values = [torch.as_tensor(sum_metrics[key]).detach().to("cpu", torch.float64)
              for key in keys]
    values += [torch.as_tensor(loss).detach().to("cpu", torch.float64),
               torch.as_tensor(response_length).detach().to("cpu", torch.float64)]
    if any(value.numel() != 1 for value in values):
        raise ValueError("Response feature observations must be scalars")
    raw = torch.stack([value.reshape(()) for value in values])
    _finite(raw, "response observations")
    if bool((raw[:-2] < 0).any()):
        raise ValueError("Response weights and counts must be nonnegative")
    length = float(raw[-1])
    if length <= 0 or length != int(length):
        raise ValueError("Response length must be a positive integer")
    for key in ("empty_blocks", "available_position_occurrences",
                "selected_position_occurrences"):
        if key in sum_metrics:
            value = torch.as_tensor(sum_metrics[key]).detach()
            _finite(value, key)
            if value.numel() != 1 or bool((value < 0).any()):
                raise ValueError(f"Invalid response count: {key}")
    raw = raw.float()
    _finite(raw, "FP32 response observations")
    available, selected, no_error = raw[:3]
    histogram = raw[3:18]
    count = no_error + histogram.sum()
    location = torch.dot(torch.arange(1, 16, dtype=torch.float32), histogram)
    location = location + 16 * no_error
    positive_loss = raw[-2].clamp_min(0)
    features = torch.stack((
        selected / (available + 1e-6),
        location / (16 * count) if count > 0 else count.new_zeros(()),
        no_error / count if count > 0 else count.new_zeros(()),
        positive_loss / (1 + positive_loss),
        raw[-1] / 2048,
    )).clamp(0, 1)
    _finite(features, "response features")
    return features


@dataclass(frozen=True)
class Allocation:
    a: float
    b: torch.Tensor
    phi_entry: torch.Tensor
    score: float

    @property
    def weights(self):
        return (0.5 + self.a, 0.5 - self.a)


def _controller_vector(value, label):
    value = torch.as_tensor(value).detach().to("cpu", torch.float64)
    if value.shape != (5,):
        raise ValueError(f"{label} must have five elements")
    _finite(value, label)
    return value


class ResponseWeightController:
    """Delayed credit state; event-boundary persistence rejects pending credit."""

    def __init__(self):
        self.phi = torch.zeros(5, dtype=torch.float64)
        self.v = 0.0
        self.n = 0
        self.windows = 0
        self.previews = 0
        self.pending = None

    @torch.no_grad()
    def allocate(self, first, second, phi_entry=None):
        phi = _controller_vector(self.phi if phi_entry is None else phi_entry, "phi")
        # Features are explicitly rounded to FP32 before entering CPU FP64.
        first = _controller_vector(torch.as_tensor(first).float(), "first features")
        second = _controller_vector(torch.as_tensor(second).float(), "second features")
        if bool(((first < 0) | (first > 1) | (second < 0) | (second > 1)).any()):
            raise ValueError("Controller features must be in [0, 1]")
        difference = first - second
        score = float(torch.dot(phi, difference))
        if not math.isfinite(score):
            raise ValueError("Nonfinite controller score")
        tangent = math.tanh(score)
        return Allocation(0.25 * tangent, 0.25 * (1 - tangent * tangent) * difference,
                          phi.clone(), score)

    @torch.no_grad()
    def update_credit(self, credit, derivative):
        """Positive credit decreases the producing window's cached direction."""
        credit = float(credit)
        derivative = _controller_vector(derivative, "cached derivative")
        if not math.isfinite(credit):
            raise ValueError("Nonfinite delayed credit")
        before = self.phi.clone()
        next_n = self.n + 1
        squared_credit = credit * credit
        next_v = 0.95 * self.v + 0.05 * squared_credit
        corrected = next_v / (1 - 0.95 ** next_n)
        if not math.isfinite(corrected):
            raise ValueError("Nonfinite credit RMS")
        raw_scaled = credit / (math.sqrt(corrected) + 1e-12)
        scaled = max(-1.0, min(1.0, raw_scaled))
        increment = -0.05 * scaled * derivative
        _finite(increment, "controller increment")
        projected = False
        after = before
        if bool((increment != 0).any()):
            after = before + increment
            norm = float(torch.linalg.vector_norm(after))
            if not math.isfinite(norm):
                raise ValueError("Nonfinite controller proposal")
            if norm > 2:
                after = after * (2 / norm)
                projected = True
        # An identically zero increment deliberately skips projection and copy.
        if after is not before:
            self.phi.copy_(after)
        self.v, self.n = next_v, next_n
        return {
            "credit": credit, "normalized_credit": scaled,
            "credit_clipped": raw_scaled != scaled, "controller_projected": projected,
            "controller_update_norm": float(torch.linalg.vector_norm(self.phi - before)),
            "phi_before": before.tolist(), "phi_after": self.phi.tolist(),
            "v": self.v, "n": self.n,
        }

    def credit(self, query, names):
        if self.pending is None:
            return None
        pending = self.pending
        if pending["names"] != tuple(names):
            raise ValueError("Delayed credit parameter names/order differ")
        value = chunked_dot(query, pending["eligibility"])
        row = self.update_credit(value, pending["b"])
        row["producing_identity"] = copy.deepcopy(pending["identity"])
        self.pending = None
        return row

    def save_pending(self, eligibility, b, identity, *, names, end_of_pass=False):
        if self.pending is not None:
            raise ValueError("Previous delayed credit was not consumed")
        names = tuple(names)
        if len(names) != len(eligibility) or len(set(names)) != len(names):
            raise ValueError("Eligibility parameter names differ")
        for value in eligibility:
            if value.dtype != torch.float32:
                raise ValueError("Eligibility must be FP32")
            _finite(value, "eligibility")
        self.pending = {
            "eligibility": eligibility, "b": _controller_vector(b, "cached derivative").clone(),
            "identity": copy.deepcopy(identity), "names": names,
        }
        self.windows += 1
        self.previews += 2
        if end_of_pass:
            self.clear_pending()

    def clear_pending(self):
        self.pending = None

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("KW-v1 recovery requires a complete event with no pending credit")
        return {
            "contract": copy.deepcopy(KW_CONTRACT), "phi": self.phi.clone(),
            "v": self.v, "n": self.n, "windows": self.windows,
            "previews": self.previews, "pending": None,
        }

    def load_state_dict(self, state):
        if self.pending is not None:
            raise ValueError("Cannot restore over a pending within-pass credit")
        expected_keys = {"contract", "phi", "v", "n", "windows", "previews", "pending"}
        if set(state) != expected_keys or state["contract"] != KW_CONTRACT:
            raise ValueError("KW-v1 checkpoint contract differs")
        if state["pending"] is not None:
            raise ValueError("KW-v1 checkpoint has pending credit outside event boundary")
        phi = state["phi"]
        if not isinstance(phi, torch.Tensor) or phi.dtype != torch.float64 or phi.device.type != "cpu":
            raise ValueError("KW-v1 checkpoint phi must be CPU FP64")
        phi = _controller_vector(phi, "checkpoint phi")
        if float(torch.linalg.vector_norm(phi)) > 2 + 1e-14:
            raise ValueError("KW-v1 checkpoint phi exceeds coefficient cap")
        v = state["v"]
        counts = [state[key] for key in ("n", "windows", "previews")]
        if not isinstance(v, float) or not math.isfinite(v) or v < 0:
            raise ValueError("Invalid KW-v1 checkpoint RMS")
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Invalid KW-v1 checkpoint counters")
        n, windows, previews = counts
        if previews != 2 * windows or n > windows or (n == 0 and v != 0):
            raise ValueError("Inconsistent KW-v1 checkpoint counters")
        self.phi.copy_(phi)
        self.v, self.n, self.windows, self.previews = v, n, windows, previews
        self.pending = None


class HalfGradientCapture:
    """Capture leaf backward contributions before native BF16 accumulation.

Begin/end once per ordinary response backward. Returning None leaves native
autograd and DDP bucket accumulation unchanged. Repeated hook invocations in
one backward are accumulated in their native dtype and observed order.
"""

    def __init__(self, named_parameters):
        self.named_parameters = tuple(named_parameters)
        self.active = None
        self.capture_host_s = 0.0
        self.handles = []
        for index, (_, parameter) in enumerate(self.named_parameters):
            self.handles.append(parameter.register_hook(self._hook(index)))

    def _hook(self, index):
        def capture(gradient):
            started = time.perf_counter()
            if self.active is None:
                raise RuntimeError("KW gradient hook outside a response capture")
            if self.active[index] is None:
                self.active[index] = gradient.detach().clone()
            else:
                self.active[index].add_(gradient.detach())
            self.capture_host_s += time.perf_counter() - started
            return None
        return capture

    def begin_response(self):
        if self.active is not None:
            raise RuntimeError("KW response capture already active")
        self.active = [None] * len(self.named_parameters)

    def end_response(self):
        if self.active is None:
            raise RuntimeError("KW response capture is not active")
        captured, self.active = self.active, None
        return captured

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = None


def clone_native_gradients(named_parameters):
    return [None if parameter.grad is None else parameter.grad.detach().clone()
            for _, parameter in named_parameters]


def _validate_allocation(a):
    a = float(a)
    if not math.isfinite(a) or not -0.25 <= a <= 0.25:
        raise ValueError("Allocation must be finite and in [-0.25, 0.25]")
    return a


@torch.no_grad()
def weighted_gradients(native, first, second, a):
    """Separate eager FP32 subtract/multiply/add, then exactly one BF16 cast.

The neutral result aliases native gradients. Missing native gradients stay
None, including when a response hook supplied no contribution.
"""
    a = _validate_allocation(a)
    if not len(native) == len(first) == len(second):
        raise ValueError("Gradient parameter lists differ")
    result = []
    for gradient, one, two in zip(native, first, second):
        for value in (gradient, one, two):
            if value is not None:
                if value.dtype != torch.bfloat16:
                    raise ValueError("KW native and half gradients must be BF16")
                _finite(value, "gradient")
        if gradient is None:
            if one is not None or two is not None:
                raise ValueError("Captured contribution without a native active gradient")
            result.append(None)
            continue
        for value in (one, two):
            if value is not None and (value.shape != gradient.shape or value.device != gradient.device):
                raise ValueError("Captured gradient shape/device differs")
        if a == 0:
            result.append(gradient)
            continue
        difference = torch.zeros_like(gradient, dtype=torch.float32) if one is None else one.float()
        if two is not None:
            difference = torch.sub(difference, two.float())
        else:
            difference = torch.sub(difference, 0.0)
        perturbation = torch.mul(difference, torch.tensor(2 * a, dtype=torch.float32,
                                                       device=gradient.device))
        weighted = torch.add(gradient.float(), perturbation).to(torch.bfloat16)
        _finite(weighted, "weighted gradient")
        result.append(weighted)
    return result


@torch.no_grad()
def apply_weighted_gradients_(named_parameters, native, first, second, a):
    """Copy only at nonzero allocation, preserving existing .grad bucket views.

If native aliases live .grad, call only after credit/previews have finished:
the final copy necessarily replaces those live values before their native clip.
"""
    named_parameters = tuple(named_parameters)
    if len(named_parameters) != len(native):
        raise ValueError("Native gradient parameter list differs")
    weighted = weighted_gradients(native, first, second, a)
    for (_, parameter), gradient, original in zip(named_parameters, weighted, native):
        if (parameter.grad is None) != (original is None):
            raise ValueError("Live native active-gradient set changed")
        if gradient is not None and float(a) != 0:
            parameter.grad.copy_(gradient)


@torch.no_grad()
def chunked_dot(left, right, *, chunk_elements=MAX_CHUNK_ELEMENTS):
    """Fixed parameter/chunk order; FP64 products and sums, CPU FP64 total."""
    if not 0 < chunk_elements <= MAX_CHUNK_ELEMENTS or len(left) != len(right):
        raise ValueError("Invalid chunk size or parameter lists")
    total = torch.zeros((), dtype=torch.float64, device="cpu")
    for one, two in zip(left, right):
        if two is not None:
            _finite(two, "dot right operand")
        if one is None:
            continue
        _finite(one, "dot left operand")
        if two is None:
            continue
        if one.shape != two.shape or one.device != two.device:
            raise ValueError("Dot operand shape/device differs")
        one, two = one.reshape(-1), two.reshape(-1)
        for start in range(0, one.numel(), chunk_elements):
            product = one[start:start + chunk_elements].double() * two[start:start + chunk_elements].double()
            total.add_(product.sum(dtype=torch.float64).cpu())
    _finite(total, "FP64 dot reduction")
    return float(total)


def gradient_norm(gradients):
    return math.sqrt(chunked_dot(gradients, gradients))


def _clone_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


class NativeAdamWPreview:
    """One reusable scratch optimizer with the native groups, dtype and dispatch.

Every apply starts from the *current* live pre-step state. The live state and
its .grad objects are never modified. Scratch parameters/state are exposed for
bounded qualification; production retains only the secant.
"""

    def __init__(self, optimizer, named_parameters, *, max_grad_norm=1.0):
        if type(optimizer) is not torch.optim.AdamW:
            raise ValueError("KW previews require native torch.optim.AdamW")
        self.live_optimizer = optimizer
        self.named_parameters = tuple(named_parameters)
        self.max_grad_norm = float(max_grad_norm)
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("Invalid preview clipping norm")
        self.live_parameters = [parameter for _, parameter in self.named_parameters]
        if len({id(p) for p in self.live_parameters}) != len(self.live_parameters):
            raise ValueError("Duplicate named trainable parameters")
        if any(p.dtype != torch.bfloat16 or not p.requires_grad for p in self.live_parameters):
            raise ValueError("Preview parameters must be trainable BF16")
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        if len(grouped) != len(self.live_parameters) or {id(p) for p in grouped} != {id(p) for p in self.live_parameters}:
            raise ValueError("Preview optimizer/trainable parameter sets differ")
        self.parameters = [torch.nn.Parameter(p.detach().clone()) for p in self.live_parameters]
        mapping = {id(live): scratch for live, scratch in zip(self.live_parameters, self.parameters)}
        groups = []
        for group in optimizer.param_groups:
            options = _clone_state({key: value for key, value in group.items() if key != "params"})
            groups.append({**options, "params": [mapping[id(p)] for p in group["params"]]})
        defaults = _clone_state(optimizer.defaults)
        # AdamW stores this inherited Adam flag, but its constructor fixes it
        # internally rather than accepting it as a keyword in PyTorch 2.9.
        defaults.pop("decoupled_weight_decay", None)
        self.optimizer = torch.optim.AdamW(groups, **defaults)

    @torch.no_grad()
    def apply(self, native, first, second, a, *, on_optimizer_step=None):
        gradients = weighted_gradients(native, first, second, a)
        if len(gradients) != len(self.parameters):
            raise ValueError("Preview gradient parameter list differs")
        self.optimizer.state.clear()
        for live, scratch, gradient in zip(self.live_parameters, self.parameters, gradients):
            _finite(live, "pre-step parameter")
            scratch.copy_(live)
            scratch.grad = None if gradient is None else gradient.detach().clone()
            if live in self.live_optimizer.state:
                state = self.live_optimizer.state[live]
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        _finite(value, "pre-step Adam state")
                self.optimizer.state[scratch] = _clone_state(state)
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.max_grad_norm, error_if_nonfinite=True,
        )
        if on_optimizer_step is not None:
            on_optimizer_step(float(a))
        self.optimizer.step()
        for parameter in self.parameters:
            _finite(parameter, "optimizer proposal")
            for value in self.optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor):
                    _finite(value, "optimizer proposal state")
        return norm.detach()


@dataclass
class SecantResult:
    eligibility: list
    stats: dict


@torch.no_grad()
def optimizer_secant(optimizer, named_parameters, native, first, second, a, *,
                     max_grad_norm=1.0, on_optimizer_step=None):
    """Exactly two sequential, isolated native optimizer applications per window."""
    a = _validate_allocation(a)
    plus = min(0.25, a + 1 / 16)
    minus = max(-0.25, a - 1 / 16)
    scratch = NativeAdamWPreview(optimizer, named_parameters, max_grad_norm=max_grad_norm)
    plus_norm = scratch.apply(native, first, second, plus, on_optimizer_step=on_optimizer_step)
    eligibility = [parameter.detach().float() for parameter in scratch.parameters]
    minus_norm = scratch.apply(native, first, second, minus, on_optimizer_step=on_optimizer_step)
    changed, count = 0, 0
    denominator = torch.tensor(plus - minus, dtype=torch.float32)
    for value, parameter in zip(eligibility, scratch.parameters):
        proposal = parameter.detach().float()
        changed += int(torch.count_nonzero(value != proposal))
        count += value.numel()
        value.sub_(proposal)
        value.div_(denominator.to(value.device))
        _finite(value, "optimizer secant")
    return SecantResult(eligibility, {
        "a_plus": plus, "a_minus": minus, "secant_denominator": plus - minus,
        "secant_norm": gradient_norm(eligibility), "zero_secant": changed == 0,
        "changed_coordinates": changed, "parameter_coordinates": count,
        "changed_coordinate_fraction": changed / count if count else 0.0,
        "plus_preclip_norm": float(plus_norm), "minus_preclip_norm": float(minus_norm),
        "optimizer_previews": 2,
    })
