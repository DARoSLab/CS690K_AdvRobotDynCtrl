"""Numerical inverse kinematics: damped least-squares Newton-Raphson.  (Given.)

This is the solver you wrote in the Jacobian homework:

    e  = pose_error(current, target)            [omega; v]
    dq = J^T (J J^T + lam^2 I)^-1 e
    q <- clip(q + dq, q_lo, q_hi)

It is the baseline the learned policy is compared against.
"""
from __future__ import annotations

import time

import numpy as np

from .robot import Arm, pose_error


class DLSIK:
    def __init__(self, arm: Arm | None = None, lam: float = 1e-2, iters: int = 200,
                 pos_tol: float = 1e-4, ori_tol: float = 1e-3):
        self.arm = arm or Arm()
        self.lam, self.iters, self.pos_tol, self.ori_tol = lam, iters, pos_tol, ori_tol

    def solve(self, q0, p_des, R_des):
        """One Newton run from q0. Returns (q, converged, iterations)."""
        a = self.arm
        q = np.clip(np.asarray(q0, float).copy(), a.q_lo, a.q_hi)
        for it in range(self.iters):
            p, R = a.fk(q)
            e = pose_error(p, R, p_des, R_des)
            if np.linalg.norm(e[3:]) < self.pos_tol and np.linalg.norm(e[:3]) < self.ori_tol:
                return q, True, it
            J = a.jacobian(q)
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + self.lam ** 2 * np.eye(6), e), a.q_lo, a.q_hi)
        return q, False, self.iters

    def solve_restarts(self, q0, p_des, R_des, restarts: int = 5, rng=None):
        """First run from q0; if it fails, retry from random configurations.
        Returns (q, converged, total_iterations, seconds)."""
        rng = rng if rng is not None else np.random.default_rng(0)
        t0 = time.perf_counter()
        best, total = None, 0
        start = np.asarray(q0, float)
        for r in range(restarts):
            q, ok, it = self.solve(start, p_des, R_des)
            total += it
            p, R = self.arm.fk(q)
            err = pose_error(p, R, p_des, R_des)
            score = np.linalg.norm(err[3:]) / 0.02 + np.linalg.norm(err[:3]) / 0.2
            if best is None or score < best[0]:
                best = (score, q)
            if ok:
                break
            start = rng.uniform(self.arm.q_lo, self.arm.q_hi)
        return best[1], ok, total, time.perf_counter() - t0
