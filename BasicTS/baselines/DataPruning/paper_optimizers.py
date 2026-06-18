import torch


class MuonWithAdamFallback(torch.optim.Optimizer):
    """Single-device Muon with AdamW fallback for non-matrix parameters.

    BasicTS ships a Muon optimizer, but that variant assumes all parameters are
    2D and uses torch.distributed collectives in every step. This local variant
    keeps the optimizer ablation scoped to DataPruning and works with the
    single-GPU BasicTS runner used in these MVP experiments.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
        backend_steps=5,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        trainable_params = [param for param in params if param.requires_grad]
        muon_params = [param for param in trainable_params if param.dim() >= 2]
        adam_params = [param for param in trainable_params if param.dim() < 2]

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "backend_steps": backend_steps,
            "betas": betas,
            "eps": eps,
        }
        param_groups = []
        if muon_params:
            param_groups.append({"params": muon_params, "mode": "muon"})
        if adam_params:
            param_groups.append({"params": adam_params, "mode": "adamw"})
        super().__init__(param_groups, defaults)

    @staticmethod
    def _zeropower_via_newtonschulz5(grad, steps, eps=1e-7):
        assert len(grad.shape) == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        update = grad.bfloat16() if grad.is_cuda else grad.float()
        update /= update.norm() + eps
        transposed = update.size(0) > update.size(1)
        if transposed:
            update = update.T
        for _ in range(steps):
            gram = update @ update.T
            update = a * update + b * gram @ update + c * gram @ gram @ update
        if transposed:
            update = update.T
        return update

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            mode = group.get("mode", "muon")
            if mode == "muon":
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        nesterov = group["nesterov"]
        backend_steps = group["backend_steps"]

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonWithAdamFallback does not support sparse gradients.")
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(grad)
            update = grad.add(buffer, alpha=momentum) if nesterov else buffer

            original_shape = update.shape
            if update.dim() > 2:
                update = update.reshape(update.shape[0], -1)
            update = self._zeropower_via_newtonschulz5(update, backend_steps)
            update *= max(1, update.size(0) / update.size(1)) ** 0.5
            update = update.reshape(original_shape).to(dtype=param.dtype)

            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)

    def _step_adamw_group(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonWithAdamFallback AdamW fallback does not support sparse gradients.")
            state = self.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)
