import torch
from torch.optim import Optimizer
import logging

logger = logging.getLogger(__name__)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive= False, **kwargs):
        self.rho = rho
        self.adaptive = adaptive
        self.defaults = kwargs

        params = list(params)
        self.base_optimizer = base_optimizer_cls(params, **kwargs)
        super().__init__(params, self.defaults)
        self.param_groups = self.base_optimizer.param_groups


    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        # logger.debug("SAM.first_step()")
        grad_norm = self._grad_norm()
        if not torch.isfinite(grad_norm) or grad_norm <= 0:
            # nothing to do; optionally clear grads
            if zero_grad:
                self.zero_grad()
            return

        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # grad cast: keep on param device, compute in fp32 for stability then cast back
                g = p.grad.detach()
                g32 = g.float()

                if self.adaptive:
                    e = (p.detach().abs().float() * g32) * scale
                else:
                    e = g32 * scale

                # cast perturbation to param dtype/device to avoid dtype upcast and extra memory
                e = e.to(device=p.device, dtype=p.dtype)

                # save & apply
                st = self.state[p]
                st["eps"] = e  # same dtype/device as param
                p.add_(e)

    @torch.no_grad()
    def restore(self):
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, None)
                if not st:
                    continue
                e = st.pop("eps", None)
                if e is not None:
                    p.sub_(e.to(device=p.device, dtype=p.dtype))
                    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
            self.restore()
            self.base_optimizer.step()
            if zero_grad: self.zero_grad()

    def step(self, *a, **k): return
    
    def zero_grad(self, set_to_none=True): self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        norms, device = [], self.param_groups[0]['params'][0].device
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad.detach().float()
                norms.append((p.detach().abs() * g).norm(2) if self.adaptive else g.norm(2))
        return torch.tensor(0.0, device=device) if not norms else torch.norm(torch.stack(norms), 2) 
    