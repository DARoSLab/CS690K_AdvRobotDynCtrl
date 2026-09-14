#!/usr/bin/env python3
"""Sanity-check the install: load the walker model and step it with random / zero torques.

    python play_random.py                  # print a few steps of rollout data
    python play_random.py --video rnd.mp4  # save a short clip
    python play_random.py --viewer         # interactive window (macOS: run with mjpython)

Run this first, before training. It uses MuJoCo directly and none of the code you
write in Parts A-C, so it works on a fresh handout: a failure here is an install
problem, not a bug in your solution.
"""
from __future__ import annotations

import argparse
import sys
import time

import mujoco
import numpy as np

from planar_walker.config import EnvConfig
from planar_walker.env import QPOS_NAMES, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--zero", action="store_true", help="apply zero torque instead of random")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--viewer", action="store_true")
    a = p.parse_args()

    cfg = EnvConfig()
    model = load_model()
    data = mujoco.MjData(model)
    z_lo, z_hi = cfg.reward.healthy_z_range
    print(f"nq = {model.nq}, nv = {model.nv}, nu = {model.nu}")
    print(f"qpos layout: {QPOS_NAMES}")
    print(f"control dt  = {cfg.control_dt:.4f} s, max episode steps = {cfg.max_episode_steps}")

    renderer = mujoco.Renderer(model, height=720, width=960) if a.video else None
    viewer = None
    if a.viewer:
        from mujoco import viewer as mj_viewer   # submodule; not loaded by `import mujoco`

        try:
            viewer = mj_viewer.launch_passive(model, data)
        except RuntimeError as e:
            if sys.platform == "darwin":
                raise SystemExit("On macOS the interactive viewer must be launched with mjpython:\n"
                                 "    mjpython " + " ".join(sys.argv)) from e
            raise

    rng = np.random.default_rng(0)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    frames = []
    for t in range(a.steps):
        data.ctrl[:] = 0.0 if a.zero else rng.uniform(-1.0, 1.0, model.nu)
        mujoco.mj_step(model, data, nstep=cfg.frame_skip)
        if t < 5:
            print(f"  t={t:3d}  x={data.qpos[0]:+.3f}  z={data.qpos[1]:+.3f}  pitch={data.qpos[2]:+.3f}")
        if renderer is not None:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render())
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(cfg.control_dt)
        if not z_lo < data.qpos[1] < z_hi:
            print(f"  torso height left the default range at t={t} (z = {data.qpos[1]:.2f} m); resetting")
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
    print(f"ran {a.steps} control steps")

    if renderer is not None:
        import imageio

        fps = int(round(1.0 / cfg.control_dt))
        imageio.mimsave(a.video, frames, fps=fps)
        renderer.close()
        print(f"wrote {a.video}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
