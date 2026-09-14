"""Actor-critic network for PPO.

============================================================================
 HW1 -- PART C : POLICY / VALUE NETWORK                  [student section]
============================================================================
`ActorCritic` below is a reference solution: two separate tanh MLPs (one for
the policy mean, one for the value), a *state-independent* log-std parameter,
and orthogonal weight init.  For the assignment, the bodies of `__init__` and
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
    # TODO (HW1 Part C): build an MLP  in_dim -> hidden... -> out_dim.
    # Use layer_init(nn.Linear(...)) for the weights; the final layer uses std=out_std.
    raise NotImplementedError("HW1 Part C: implement _build_mlp")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (64, 64)):
        super().__init__()
        # TODO (HW1 Part C): create the actor and critic networks and the log-std.
        # Needed attributes (used elsewhere in this class):
        #   self.actor_mean : nn.Module  obs -> action mean
        #   self.critic     : nn.Module  obs -> state value (shape [...,1])
        #   self.actor_logstd : nn.Parameter of shape (1, act_dim)
        raise NotImplementedError("HW1 Part C: implement ActorCritic.__init__")

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
