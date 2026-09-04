#!/usr/bin/env python3
"""Train a walking policy with PPO.

Examples
--------
    python train.py                              # default 1.2M steps (~4-6 min, laptop CPU)
    python train.py --total-timesteps 300000     # quick smoke test
    python train.py --num-envs 4 --seed 1
    python train.py --target-speed 1.5 --ctrl-cost-weight 5e-4

Outputs
-------
    checkpoints/policy.pt         best-so-far (by eval return)
    checkpoints/policy_final.pt   policy at the end of training
Then visualise with:
    python evaluate.py --checkpoint checkpoints/policy.pt --video walk.mp4
"""
from __future__ import annotations

import argparse

from planar_walker.config import EnvConfig, PPOConfig, RewardConfig
from planar_walker.ppo import train


def build_configs(a: argparse.Namespace):
    reward = RewardConfig(
        target_speed=a.target_speed,
        forward_weight=a.forward_weight,
        alive_bonus=a.alive_bonus,
        ctrl_cost_weight=a.ctrl_cost_weight,
    )
    env_cfg = EnvConfig(
        frame_skip=a.frame_skip,
        episode_seconds=a.episode_seconds,
        reward=reward,
    )
    ppo_cfg = PPOConfig(
        num_envs=a.num_envs,
        num_steps=a.num_steps,
        total_timesteps=a.total_timesteps,
        learning_rate=a.learning_rate,
        update_epochs=a.update_epochs,
        num_minibatches=a.num_minibatches,
        clip_coef=a.clip_coef,
        ent_coef=a.ent_coef,
        seed=a.seed,
        torch_threads=a.torch_threads,
        hidden_sizes=tuple(a.hidden_sizes),
        eval_every_updates=a.eval_every_updates,
    )
    return env_cfg, ppo_cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # reward
    p.add_argument("--target-speed", type=float, default=1.0)
    p.add_argument("--forward-weight", type=float, default=1.0)
    p.add_argument("--alive-bonus", type=float, default=1.0)
    p.add_argument("--ctrl-cost-weight", type=float, default=1e-3)
    # env
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--episode-seconds", type=float, default=8.0)
    # ppo
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=512)
    p.add_argument("--total-timesteps", type=int, default=1_200_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    p.add_argument("--eval-every-updates", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default="checkpoints/policy.pt")
    a = p.parse_args()

    env_cfg, ppo_cfg = build_configs(a)
    print("Env config :", env_cfg)
    print("PPO config :", ppo_cfg)
    print(f"batch={ppo_cfg.batch_size}  minibatch={ppo_cfg.minibatch_size}  "
          f"updates={ppo_cfg.num_updates}  control_dt={env_cfg.control_dt:.4f}s  "
          f"max_ep_steps={env_cfg.max_episode_steps}")
    train(ppo_cfg, env_cfg, ckpt_path=a.checkpoint)


if __name__ == "__main__":
    main()
