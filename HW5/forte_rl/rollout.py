"""Run a controller on a set of targets and measure what happened.  (Given.)

A "controller" is anything with the signature  controller(env) -> action  that is
called once per decision, after `env.reset`. Two are provided:

    PolicyController(policy)     the learned policy: action = policy(obs)
    IKController(ik)             numerical IK: solve once at the start of the
                                 episode, then move the set-point toward the
                                 solution at the same max rate the policy has

Both drive the SAME environment, with the SAME joint PD underneath, so any
difference in the numbers comes from how the joint targets were chosen.
"""
from __future__ import annotations

import time

import numpy as np

from .robot import CollisionChecker
from .task import is_success


class PolicyController:
    def __init__(self, policy):
        self.policy = policy
        self.compute_s = 0.0

    def start(self, env, obs):
        self.obs = obs

    def __call__(self, env):
        t0 = time.perf_counter()
        a = self.policy(self.obs)
        self.compute_s += time.perf_counter() - t0
        return a


class IKController:
    def __init__(self, ik, restarts: int = 5, seed: int = 0):
        self.ik, self.restarts = ik, restarts
        self.rng = np.random.default_rng(seed)
        self.compute_s = 0.0
        self.converged = None

    def start(self, env, obs):
        q, ok, it, sec = self.ik.solve_restarts(env.d.qpos.copy(), env.p_des, env.R_des,
                                                restarts=self.restarts, rng=self.rng)
        self.q_goal, self.converged = q, ok
        self.compute_s += sec

    def __call__(self, env):
        return np.clip((self.q_goal - env.q_cmd) / env.cfg.max_dq, -1.0, 1.0)


def evaluate(controller, env, targets, check_collisions: bool = False, record: bool = False):
    """Run one episode per target. Returns a dict of per-target arrays:

    success, pos_err, ori_err   final result (success = inside both tolerances)
    steps                       decisions until success (max_steps if never)
    path_len                    joint-space path length sum |dq|            [rad]
    peak_tau                    largest |tau| / torque limit over the episode
    collided                    did any visited configuration touch the floor or itself
    compute_ms                  controller computation time per episode       [ms]
    """
    P, R = targets
    cc = CollisionChecker() if check_collisions else None
    out = {k: [] for k in ("success", "pos_err", "ori_err", "steps", "path_len",
                           "peak_tau", "collided", "compute_ms")}
    traj = []
    for i in range(len(P)):
        obs, _ = env.reset(seed=i, options={"target": (P[i], R[i])})
        c0 = controller.compute_s
        controller.start(env, obs)
        q_prev = env.d.qpos.copy(); path = 0.0; peak = 0.0; hit = False; qs = [q_prev.copy()]
        steps_to_success = env.cfg.max_steps
        for k in range(env.cfg.max_steps):
            if hasattr(controller, "obs"):           # controllers that read the observation
                controller.obs = obs
            obs, _, term, trunc, info = env.step(controller(env))
            q = env.d.qpos.copy()
            path += float(np.abs(q - q_prev).sum()); q_prev = q
            peak = max(peak, float(np.max(np.abs(env.d.ctrl) / env.tau_lim)))
            if cc is not None and not hit:
                hit = cc.in_collision(q)
            if record:
                qs.append(q)
            if info["is_success"] and steps_to_success == env.cfg.max_steps:
                steps_to_success = k + 1
            if term or trunc:
                break
        out["success"].append(float(is_success(info["pos_err"], info["ori_err"])))
        out["pos_err"].append(info["pos_err"]); out["ori_err"].append(info["ori_err"])
        out["steps"].append(steps_to_success); out["path_len"].append(path)
        out["peak_tau"].append(peak); out["collided"].append(float(hit))
        out["compute_ms"].append(1000 * (controller.compute_s - c0))
        if record:
            traj.append(np.array(qs))
    res = {k: np.array(v) for k, v in out.items()}
    if record:
        res["traj"] = traj
    return res


def summarize(res: dict) -> dict:
    """One-line summary of an `evaluate` result."""
    s = res["success"] > 0.5
    return {
        "success %": 100 * s.mean(),
        "pos med [mm]": 1000 * np.median(res["pos_err"]),
        "pos p95 [mm]": 1000 * np.percentile(res["pos_err"], 95),
        "ori med [rad]": np.median(res["ori_err"]),
        "ori p95 [rad]": np.percentile(res["ori_err"], 95),
        "steps (succ.)": float(np.mean(res["steps"][s])) if s.any() else float("nan"),
        "path [rad]": float(np.mean(res["path_len"])),
        "peak tau": float(np.mean(res["peak_tau"])),
        "collided %": 100 * res["collided"].mean(),
        "compute [ms]": float(np.mean(res["compute_ms"])),
    }
