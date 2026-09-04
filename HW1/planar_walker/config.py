"""Configuration dataclasses for the planar-walker HW.

Everything a student might want to tune lives here so the training script stays
readable.  All values are plain Python; feel free to override them from the
command line (see ``train.py``) or from the notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class RewardConfig:
    """Weights for the reward terms in :func:`planar_walker.reward.compute_reward`.

    HW1 asks you to *design* the reward.  These are the reference weights that
    produce a clean walking gait; change them and see what happens.
    """
    target_speed: float = 1.0          # [m/s] desired forward (world +x) speed
    forward_weight: float = 1.0        # reward for matching target_speed
    alive_bonus: float = 1.0           # per-step bonus for staying "healthy"
    ctrl_cost_weight: float = 1e-3     # penalty on sum(action^2)
    # "healthy" (non-terminal) region for the torso:
    healthy_z_range: Tuple[float, float] = (0.8, 2.0)     # torso height [m]
    healthy_pitch_range: Tuple[float, float] = (-1.0, 1.0)  # torso pitch [rad]


@dataclass
class EnvConfig:
    frame_skip: int = 4                # sim steps per env.step  -> control dt = 0.008 s
    episode_seconds: float = 8.0       # truncation horizon
    ctrl_range: float = 1.0            # actions are clipped to [-1, 1]
    obs_clip: float = 10.0             # observations clipped to +-obs_clip after normalization
    reset_noise_scale: float = 5e-3    # uniform noise added to qpos/qvel on reset
    exclude_current_positions: bool = True  # drop absolute x from the observation
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def control_dt(self) -> float:
        # timestep in the XML is 0.002 s
        return 0.002 * self.frame_skip

    @property
    def max_episode_steps(self) -> int:
        return int(round(self.episode_seconds / self.control_dt))


@dataclass
class PPOConfig:
    # --- rollout / batch ---
    num_envs: int = 8
    num_steps: int = 512               # per-env rollout length -> batch = num_envs * num_steps
    total_timesteps: int = 1_200_000

    # --- optimization ---
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    num_minibatches: int = 32
    max_grad_norm: float = 0.5

    # --- PPO objective ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    target_kl: float | None = 0.03     # early-stop an update if KL exceeds this

    # --- normalization ---
    normalize_obs: bool = True
    normalize_reward: bool = True

    # --- network ---
    hidden_sizes: Tuple[int, ...] = (64, 64)

    # --- bookkeeping ---
    seed: int = 0
    torch_threads: int = 4             # keep CPU usage friendly on a laptop
    eval_every_updates: int = 20
    log_every_updates: int = 1

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size


def to_dict(cfg) -> dict:
    return asdict(cfg)
