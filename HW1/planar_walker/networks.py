"""Actor-critic network for PPO.

============================================================================
 HW1 -- PART C : POLICY / VALUE NETWORK                  [student section]
============================================================================
`ActorCritic` below is a reference solution: two separate tanh MLPs (one for
the policy mean, one for the value), a *state-independent* log-std parameter,
and orthogonal weight init.  For the assignment, the body of `__init__` and
`_build_mlp` are removed and you build the networks.

Questions to explore:
  * Why a diagonal Gaussian policy for continuous torque control?
  * Why is the log-std often made state-independent for locomotion?
  * What changes if the actor and critic share a trunk?
  * How much does width / depth / activation actually matter here?
============================================================================
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
    # === HW SOLUTION START: networks.build_mlp ===
    layers = []
    last = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
        last = h
    layers += [layer_init(nn.Linear(last, out_dim), std=out_std)]
    return nn.Sequential(*layers)
    # === HW SOLUTION END: networks.build_mlp ===


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (64, 64)):
        super().__init__()
        # === HW SOLUTION START: networks.actor_critic_init ===
        self.actor_mean = _build_mlp(obs_dim, hidden, act_dim, out_std=0.01)
        self.critic = _build_mlp(obs_dim, hidden, 1, out_std=1.0)
        # state-independent log std, initialised so std ~= 1.0
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        # === HW SOLUTION END: networks.actor_critic_init ===

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
