"""Direct BF16 AdamW for the single-process, NO_SHARD online OSD recipe."""

import torch


class DirectBF16AdamW:
    """Keep parameters and Adam moments in BF16, without master weights.

    TrainerCore normalizes accumulated gradients before calling ``step``. This
    adapter then clips once, applies ordinary PyTorch AdamW, and clears gradients.
    Learning rate stays constant; there is no scheduler or optimizer offload.
    """

    checks_loss_denominator = False

    def __init__(self, model, *, learning_rate=3e-4, weight_decay=0.01,
                 max_grad_norm=1.0):
        named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not named_params:
            raise ValueError("DirectBF16AdamW requires trainable draft parameters")
        wrong_dtype = [name for name, p in named_params if p.dtype != torch.bfloat16]
        if wrong_dtype:
            raise ValueError(f"DirectBF16AdamW requires BF16 parameters: {wrong_dtype}")
        self.model_params = [p for _, p in named_params]
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = torch.optim.AdamW(
            self.model_params, lr=learning_rate, weight_decay=weight_decay,
        )
        self.recipe = {
            "type": "direct_bf16_adamw", "version": 1,
            "max_grad_norm": self.max_grad_norm,
            "parameters": tuple((name, tuple(p.shape), str(p.dtype)) for name, p in named_params),
            "optimizer_options": self._options(self.optimizer.param_groups[0]),
        }

    @staticmethod
    def _options(group):
        return {name: value for name, value in group.items() if name != "params"}

    def step(self):
        """Return the pre-clip norm after one AdamW step and gradient clearing."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model_params, self.max_grad_norm, error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.zero_grad()
        return grad_norm.detach()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {
            "recipe": {**self.recipe, "optimizer_options": dict(self.recipe["optimizer_options"])},
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        """Reject a different numerical recipe before changing optimizer state."""
        if state.get("recipe") != self.recipe:
            raise ValueError("Checkpoint direct BF16 AdamW recipe differs")
        saved = state["optimizer_state_dict"]
        groups = saved["param_groups"]
        if (len(groups) != 1 or len(groups[0]["params"]) != len(self.model_params)
                or self._options(groups[0]) != self.recipe["optimizer_options"]):
            raise ValueError("Checkpoint direct BF16 AdamW parameter group differs")
        for saved_id, param in zip(groups[0]["params"], self.model_params):
            moments = saved["state"].get(saved_id, {})
            for name in ("exp_avg", "exp_avg_sq"):
                if name in moments and (moments[name].dtype != torch.bfloat16
                                        or moments[name].shape != param.shape):
                    raise ValueError(f"Checkpoint BF16 AdamW {name} dtype or shape differs")
        self.optimizer.load_state_dict(saved)
