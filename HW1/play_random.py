#!/usr/bin/env python3
"""Sanity-check the environment with random / zero actions.

    python play_random.py                 # print a few steps of rollout data
    python play_random.py --video rnd.mp4  # save a short clip
    python play_random.py --viewer         # interactive window (needs a display)

Use this first, before training, to confirm MuJoCo is installed and the model
loads on your machine.
"""
from __future__ import annotations

import argparse

import numpy as np

from planar_walker.env import PlanarWalkerEnv, QPOS_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--zero", action="store_true", help="apply zero torque instead of random")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--viewer", action="store_true")
    a = p.parse_args()

    render_mode = "rgb_array" if a.video else ("human" if a.viewer else None)
    env = PlanarWalkerEnv(render_mode=render_mode)
    print(f"obs dim = {env.observation_space.shape[0]}, act dim = {env.action_space.shape[0]}")
    print(f"qpos layout: {QPOS_NAMES}")
    print(f"control dt  = {env.cfg.control_dt:.4f} s, max episode steps = {env.cfg.max_episode_steps}")

    obs, _ = env.reset(seed=0)
    frames, total_r = [], 0.0
    for t in range(a.steps):
        action = np.zeros(env.action_space.shape) if a.zero else env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        if a.video:
            frames.append(env.render())
        if t < 5:
            print(f"  t={t:3d}  r={r:+.3f}  x={info['x_position']:+.3f}  "
                  f"healthy={info['is_healthy']}  vx={info['x_velocity']:+.2f}")
        if term or trunc:
            print(f"  episode ended at t={t} (terminated={term}, truncated={trunc})")
            obs, _ = env.reset()
    print(f"total reward over {a.steps} steps: {total_r:.2f}")

    if a.video:
        import imageio

        fps = int(round(1.0 / env.cfg.control_dt))
        imageio.mimsave(a.video, frames, fps=fps)
        print(f"wrote {a.video}")
    env.close()


if __name__ == "__main__":
    main()
