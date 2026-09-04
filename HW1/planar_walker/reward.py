"""Reward and termination logic for the planar walker.

============================================================================
 HW1  -- PART B : REWARD DESIGN                          [student section]
============================================================================
`compute_reward` and `is_healthy` below contain a *reference* solution that
produces a stable ~1 m/s walking gait.  For the assignment these bodies are
replaced by `raise NotImplementedError(...)` and you design your own.

Things worth thinking about:
  * What makes forward progress rewarding without rewarding "diving"?
  * Why do we need an alive bonus AND a termination condition?
  * What does the control-cost term trade off against?
  * Should the reward be a function of velocity or of position change?

`info` returned by the env breaks the reward into its named components so you
can plot each term over training.
============================================================================
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import RewardConfig


def is_healthy(qpos: np.ndarray, cfg: RewardConfig) -> bool:
    """Return True while the torso is in a recoverable state.

    qpos ordering: [rootx, rootz, rooty, hip_R, knee_R, ankle_R,
                    hip_L, knee_L, ankle_L].
    """
    # TODO (HW1 Part B): return True while the torso is in a recoverable state.
    # qpos = [rootx, rootz, rooty, hip_R, knee_R, ankle_R, hip_L, knee_L, ankle_L]
    raise NotImplementedError("HW1 Part B: implement is_healthy")


def compute_reward(
    qpos_before: np.ndarray,
    qpos_after: np.ndarray,
    qvel_after: np.ndarray,
    action: np.ndarray,
    control_dt: float,
    cfg: RewardConfig,
) -> Tuple[float, Dict[str, float]]:
    """Return (reward, info) for a single control step.

    Parameters
    ----------
    qpos_before, qpos_after : generalized positions before/after the step
    qvel_after              : generalized velocities after the step
    action                  : the (already clipped) action in [-1, 1]^6
    control_dt              : wall-clock seconds represented by one control step
    cfg                     : reward weights (see config.RewardConfig)
    """
    # TODO (HW1 Part B): design the reward.
    # Return (float reward, dict info). `info` should break the reward into
    # named components, e.g. {'reward_forward':..., 'reward_alive':..., 'reward_ctrl':...}.
    raise NotImplementedError("HW1 Part B: implement compute_reward")
