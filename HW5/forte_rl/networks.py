"""Actor-critic network for PPO.  (Given.)

Two separate tanh MLPs, one for the policy mean and one for the value, a
state-independent log-std, and orthogonal weight initialization.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _build_mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, out_std: float) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
        last = h
    layers += [layer_init(nn.Linear(last, out_dim), std=out_std)]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (64, 64),
                 logstd_init: float = 0.0):
        super().__init__()
        self.actor_mean = _build_mlp(obs_dim, hidden, act_dim, out_std=0.01)
        self.critic = _build_mlp(obs_dim, hidden, 1, out_std=1.0)
        # State-independent log std. `logstd_init` sets how big the policy's
        # exploration steps are AT THE START, and it matters enormously for a
        # precision task: exp(0)=1.0 means every sampled action is full-scale,
        # so the agent only ever experiences flailing and can never discover
        # that small, careful moves pay. exp(-1.5)~0.22 lets it explore finely
        # while still being free to output a full-scale action when far away.
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(logstd_init)))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """Return (action, log_prob, entropy, value)."""
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy action (the distribution mean) -- used at evaluation time."""
        return self.actor_mean(obs)
