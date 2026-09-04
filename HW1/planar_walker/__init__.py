"""Planar walking-robot HW package.

Modules
-------
config    : all tunable hyper-parameters (dataclasses)
env       : PlanarWalkerEnv  -- self-contained Gymnasium environment
reward    : compute_reward / is_healthy        [HW student section B]
networks  : ActorCritic MLP                    [HW student section C]
ppo       : compact PPO trainer + evaluation + checkpoint IO
utils     : seeding, running normalizers, timing
"""
from .config import EnvConfig, PPOConfig, RewardConfig
from .env import PlanarWalkerEnv, make_env, QPOS_NAMES

__all__ = [
    "EnvConfig",
    "PPOConfig",
    "RewardConfig",
    "PlanarWalkerEnv",
    "make_env",
    "QPOS_NAMES",
]
