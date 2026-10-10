"""Online adapters for the dissipative head (the conservative core stays frozen)."""
import torch
from .timereversal import design, step_terms, RLS


class NoAdapt:
    def __init__(self, model):
        self.model = model

    def update(self, traj):
        pass


class TraceAdapter:
    """Recursive parity-separated regression. Odd features update (dv, dc); even features (incl. u) absorb
    conservative / actuator error so it cannot alias into dissipation. Only the head is written back."""

    def __init__(self, model, forget=0.995, p0=1e2, p0_even=1.0, forget_fast=None, fit_gain=True, fit_even=True):
        """forget_fast: if given, forgetting used while `fast=True` (alarm active); `forget` is used otherwise.
        Long memory when healthy keeps variance low; short memory after an alarm gives fast tracking."""
        self.model = model
        self.lam_slow, self.lam_fast, self.fast = forget, (forget if forget_fast is None else forget_fast), False
        self.gain_ref = model.head.gain.detach().clone()
        n = model.n
        d = torch.zeros(1, 2, n)  # dummy traj to size the design matrix
        dummy = dict(q=torch.zeros(1, 2, n), p=torch.zeros(1, 2, n), u=torch.zeros(1, 1, n), dt=0.01)
        _, _, self.idx = design(model, dummy, gain_ref=self.gain_ref)
        self.fit_gain, self.fit_even = fit_gain, fit_even
        P = sum(len(v) for v in self.idx.values())
        keep = self.idx["dv"] + self.idx["dc"] + (self.idx["dgain"] if fit_gain else []) + (self.idx["even"] if fit_even else [])
        self.cols = sorted(keep)                       # columns actually estimated
        th0 = torch.zeros(P)
        th0[self.idx["dv"]] = model.head.dv.detach()
        th0[self.idx["dc"]] = model.head.dc.detach()
        self.rls = RLS(len(self.cols), forget=forget, p0=p0, theta0=th0[self.cols])
        d = torch.full((P,), p0)
        d[self.idx["even"]] = p0_even  # tight prior on nuisance (conservative) terms: less variance in healthy operation
        self.rls.P = torch.diag(d[self.cols])
        self.pos = {c: i for i, c in enumerate(self.cols)}

    def set_fast(self, flag):
        self.fast = bool(flag)

    def update(self, traj):
        with torch.no_grad():
            X, y, _ = design(self.model, traj, gain_ref=self.gain_ref)
            self.rls.lam = self.lam_fast if self.fast else self.lam_slow
            self.rls.update(X[:, self.cols], y)
            th = self.rls.theta
            g = lambda names: torch.stack([th[self.pos[c]] for c in names])
            self.model.head.set_values(dv=g(self.idx["dv"]).clamp_min(1e-4), dc=g(self.idx["dc"]).clamp_min(1e-4),
                                       gain=(self.gain_ref + g(self.idx["dgain"])).clamp_min(0.3) if self.fit_gain else None)



class SgdHeadAdapter:
    """Baseline: plain SGD/Adam on the head (dissipation + gain) minimising the forward OM residual.
    Same parameters and per-update step budget as TRACE, but no parity separation."""

    def __init__(self, model, lr=0.02, steps=5):
        self.model, self.steps = model, steps
        self.opt = torch.optim.Adam(model.dissipative_parameters(), lr=lr)

    def update(self, traj):
        for _ in range(self.steps):
            a, qd, _ = step_terms(self.model, traj)
            loss = ((a + self.model.friction_force(qd)) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
