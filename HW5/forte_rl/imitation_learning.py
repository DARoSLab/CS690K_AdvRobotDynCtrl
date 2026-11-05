"""Supervised learning helpers for the imitation-learning part.  (Given.)

Two uses:

  fit_mlp(X, Y)              plain regression, X -> Y, with a mean-squared-error
                             loss. Used to learn  target pose -> joint angles.

  behavior_cloning(O, A)     train the POLICY network's mean to reproduce an
                             expert's actions A in observations O. Returns an
                             `init` dict that `ppo.train(..., init=...)` accepts,
                             so reinforcement learning can start from it.

The networks and the training loops are given; deciding WHAT data to train on is
the homework.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .networks import _build_mlp
from .ppo import Policy
from .utils import RunningMeanStd


def _train(net, X, Y, epochs, lr, batch, seed, verbose, label):
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32); Y = torch.as_tensor(Y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(X); losses = []
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step(); losses.append(tot / n)
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  {label} epoch {ep + 1:4d}/{epochs}  mse {losses[-1]:.5f}")
    return losses


class Regressor:
    """A trained X -> Y network with input/output standardization."""

    def __init__(self, net, xm, xs, ym, ys):
        self.net, self.xm, self.xs, self.ym, self.ys = net, xm, xs, ym, ys

    @torch.no_grad()
    def __call__(self, X):
        X = np.atleast_2d(X)
        Z = self.net(torch.as_tensor((X - self.xm) / self.xs, dtype=torch.float32)).numpy()
        return Z * self.ys + self.ym


def fit_mlp(X, Y, hidden=(256, 256), epochs=300, lr=1e-3, batch=256, seed=0, verbose=True):
    """Fit Y = f(X) by minimizing mean-squared error. Returns (regressor, losses)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    xm, xs = X.mean(0), X.std(0) + 1e-8
    ym, ys = Y.mean(0), Y.std(0) + 1e-8
    net = _build_mlp(X.shape[1], tuple(hidden), Y.shape[1], out_std=1.0)
    losses = _train(net, (X - xm) / xs, (Y - ym) / ys, epochs, lr, batch, seed, verbose, "fit")
    return Regressor(net, xm, xs, ym, ys), losses


def behavior_cloning(O, A, hidden=(256, 256), epochs=100, lr=1e-3, batch=512, seed=0,
                     logstd=-2.0, obs_clip=10.0, verbose=True):
    """Train a policy mean to imitate expert actions.

    O : (N, obs_dim) observations visited by the expert
    A : (N, 7) the expert's actions in [-1, 1]
    Returns (policy, init, losses). `policy(obs)` runs the cloned network;
    `init` can be passed to ppo.train to continue with reinforcement learning.
    `logstd` is the exploration noise RL starts with: smaller than for a random
    network, because the cloned policy is already roughly right.
    """
    O, A = np.asarray(O, np.float64), np.asarray(A, np.float64)
    rms = RunningMeanStd(shape=(O.shape[1],)); rms.update(O)
    On = rms.normalize(O, clip=obs_clip)
    net = _build_mlp(O.shape[1], tuple(hidden), A.shape[1], out_std=0.01)
    losses = _train(net, On, A, epochs, lr, batch, seed, verbose, "BC")

    class _Agent(nn.Module):              # minimal stand-in so `Policy` can run it
        def __init__(self, m):
            super().__init__(); self.actor_mean = m
        @torch.no_grad()
        def act_deterministic(self, o):
            return self.actor_mean(o)

    policy = Policy(_Agent(net), rms, obs_clip)
    init = {"actor_mean": net.state_dict(), "obs_rms": rms, "logstd": logstd}
    return policy, init, losses
