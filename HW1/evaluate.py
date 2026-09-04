#!/usr/bin/env python3
"""Load a trained policy, roll it out, print metrics, and optionally save a video.

Examples
--------
    python evaluate.py --checkpoint checkpoints/policy.pt --episodes 10
    python evaluate.py --checkpoint checkpoints/policy.pt --video walk.mp4
    python evaluate.py --checkpoint checkpoints/policy.pt --viewer     # interactive window
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from planar_walker.env import PlanarWalkerEnv
from planar_walker.ppo import load_checkpoint


def rollout(agent, obs_rms, env_cfg, episodes, seed, render, obs_clip):
    env = PlanarWalkerEnv(cfg=env_cfg, render_mode="rgb_array" if render else None)
    frames, rets, lens, dists = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, n = False, 0.0, 0
        while not done:
            o = obs_rms.normalize(obs[None], clip=obs_clip) if obs_rms is not None else obs[None]
            a = agent.act_deterministic(torch.as_tensor(o, dtype=torch.float32)).numpy()[0]
            obs, r, term, trunc, info = env.step(a)
            ret += r
            n += 1
            done = term or trunc
            if render and ep == 0:
                frames.append(env.render())
        rets.append(ret)
        lens.append(n)
        dists.append(info["x_position"])
    env.close()
    return frames, np.array(rets), np.array(lens), np.array(dists)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--video", type=str, default=None, help="path to .mp4 or .gif to write")
    p.add_argument("--viewer", action="store_true", help="open an interactive MuJoCo window")
    a = p.parse_args()

    agent, obs_rms, env_cfg, _ = load_checkpoint(a.checkpoint)
    print(f"loaded {a.checkpoint}")

    if a.viewer:
        _run_viewer(agent, obs_rms, env_cfg)
        return

    frames, rets, lens, dists = rollout(
        agent, obs_rms, env_cfg, a.episodes, a.seed,
        render=a.video is not None, obs_clip=env_cfg.obs_clip,
    )
    dt = env_cfg.control_dt
    print(f"return  : {rets.mean():8.1f} +/- {rets.std():.1f}")
    print(f"length  : {lens.mean():8.1f} steps  ({lens.mean() * dt:.1f} s)")
    print(f"distance: {dists.mean():8.2f} m")
    print(f"speed   : {(dists / (lens * dt)).mean():8.2f} m/s")

    if a.video:
        import imageio

        fps = int(round(1.0 / dt))
        imageio.mimsave(a.video, frames, fps=fps)
        print(f"wrote {a.video}  ({len(frames)} frames @ {fps} fps)")


def _run_viewer(agent, obs_rms, env_cfg):
    import mujoco
    import mujoco.viewer

    env = PlanarWalkerEnv(cfg=env_cfg)
    obs, _ = env.reset(seed=0)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        import time

        while viewer.is_running():
            o = obs_rms.normalize(obs[None], clip=env_cfg.obs_clip) if obs_rms is not None else obs[None]
            a = agent.act_deterministic(torch.as_tensor(o, dtype=torch.float32)).numpy()[0]
            obs, r, term, trunc, info = env.step(a)
            viewer.sync()
            time.sleep(env_cfg.control_dt)
            if term or trunc:
                obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
