"""Reaching environment: a policy commands joint TARGETS, a joint PD tracks them.  (Given.)

One environment step (one decision of the policy) lasts `control_dt` = 20 ms:

    1. the policy sees   obs = [cos q, sin q, 0.1*qdot, tool pose, target pose,
                                pose error, time left]
    2. it outputs        a in [-1, 1]^7
    3. the joint set-point moves  q_cmd <- q_cmd + max_dq * a
       (kept within `max_lead` of the actual q and inside the joint limits)
    4. for 20 ms the low-level controller runs at the physics rate:
           tau = Kp (q_cmd - q) - Kd qdot + (C + G)       (clipped to the motor limits)
       i.e. the joint PD with gravity and Coriolis compensation from the
       operational-space-control homework.

So the policy never sets joint angles directly: it moves a set-point, and the arm
follows it through real dynamics (inertia, torque limits, lag).

Each episode starts at `ready_pose` (optionally perturbed), commands one target and
always lasts `max_steps` decisions (0.6 s): the policy has to REACH the target and then
HOLD it until the end. Success is judged at the last step.

Targets come from one of two sources, chosen when the env is built:
    targets=(P, R)       a fixed set: P is (N, 3), R is (N, 3, 3); one is drawn per episode
    target_sampler=f     f(rng) -> (p, R), called at every reset
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install gymnasium") from e

from .robot import MODEL_PATH, log_SO3, rot6d
from .task import is_success


@dataclass
class EnvConfig:
    control_dt: float = 0.02         # [s]  one policy decision
    physics_dt: float = 0.005        # [s]  low-level PD / physics rate (200 Hz)
    max_dq: float = 0.20             # [rad] largest set-point move per decision
    max_lead: float = 0.5            # [rad] how far the set-point may run ahead of the arm
    max_steps: int = 30              # decisions per episode (0.6 s)
    pd_omega: float = 30.0           # [rad/s] PD bandwidth per joint (critically damped)
    init_noise: float = 0.0          # [rad] uniform noise on the start configuration
    terminate_on_success: bool = False   # end the episode at the first success (not used in the homework)
    obs_clip: float = 10.0


class ForteReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_fn: Callable, targets=None, target_sampler=None,
                 cfg: Optional[EnvConfig] = None):
        assert (targets is None) != (target_sampler is None), "give targets OR target_sampler"
        self.cfg = cfg or EnvConfig()
        self.reward_fn = reward_fn
        self.targets = targets
        self.target_sampler = target_sampler

        self.m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.m.opt.timestep = self.cfg.physics_dt
        self.d = mujoco.MjData(self.m)
        self.nq = self.m.nq
        self.ee = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "ee_tip")
        self.q_lo = self.m.jnt_range[:, 0].copy()
        self.q_hi = self.m.jnt_range[:, 1].copy()
        self.q_home = self.m.key_qpos[1].copy()
        self.tau_lim = self.m.actuator_ctrlrange[:, 1].copy()
        self.n_sub = int(round(self.cfg.control_dt / self.cfg.physics_dt))

        # Joint PD gains: critically damped at `pd_omega` for each joint's own
        # inertia (the diagonal of M at ready_pose), so every joint responds alike.
        self.d.qpos[:] = self.q_home
        mujoco.mj_forward(self.m, self.d)
        M = np.zeros((self.nq, self.nq))
        mujoco.mj_fullM(self.m, M, self.d.qM) if hasattr(self.d, "qM") else mujoco.mj_fullM(self.m, self.d, M)
        I = np.diag(M).copy()
        self.Kp = I * self.cfg.pd_omega ** 2
        self.Kd = 2.0 * I * self.cfg.pd_omega

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nq,), dtype=np.float32)
        self.p_des, self.R_des = np.zeros(3), np.eye(3)
        self.q_cmd = self.q_home.copy()
        self._t = 0
        self._update_pose()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self._obs().shape, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _update_pose(self):
        mujoco.mj_kinematics(self.m, self.d)
        self.p = self.d.site_xpos[self.ee].copy()
        self.R = self.d.site_xmat[self.ee].reshape(3, 3).copy()
        self.pos_err = self.p_des - self.p
        self.ori_err = log_SO3(self.R_des @ self.R.T)

    def _obs(self) -> np.ndarray:
        q = self.d.qpos
        return np.concatenate([
            np.cos(q), np.sin(q),            # 14  joint angles, continuous across +-pi
            0.1 * self.d.qvel,               #  7  joint velocities (the arm is still moving)
            self.q_cmd - self.d.qpos,        #  7  where the set-point is, relative to the arm
            self.p, rot6d(self.R),           #  9  current tool pose
            self.p_des, rot6d(self.R_des),   #  9  target tool pose
            self.ori_err, self.pos_err,      #  6  pose error [omega; v]
            [self._t / self.cfg.max_steps],  #  1  how much time is left
        ]).astype(np.float32)

    def _draw_target(self, options):
        if options is not None and "target" in options:
            return options["target"]
        if self.targets is not None:
            P, R = self.targets
            if options is not None and "target_index" in options:
                i = int(options["target_index"]) % len(P)
            else:
                i = int(self.np_random.integers(len(P)))
            return P[i], R[i]
        return self.target_sampler(self.np_random)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        q0 = self.q_home.copy()
        if self.cfg.init_noise > 0:
            q0 = np.clip(q0 + self.np_random.uniform(-1, 1, self.nq) * self.cfg.init_noise,
                         self.q_lo, self.q_hi)
        if options is not None and "q0" in options:
            q0 = np.asarray(options["q0"], float)
        self.d.qpos[:] = q0
        self.d.qvel[:] = 0.0
        self.d.ctrl[:] = 0.0
        self.q_cmd = q0.copy()
        p, R = self._draw_target(options)
        self.p_des, self.R_des = np.asarray(p, float).copy(), np.asarray(R, float).copy()
        self._t = 0
        mujoco.mj_forward(self.m, self.d)
        self._update_pose()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # The set-point INTEGRATES: each decision moves it by at most max_dq, and it
        # may run at most `max_lead` ahead of the arm (no wind-up while the PD lags).
        q_cmd = self.q_cmd + self.cfg.max_dq * a
        q_cmd = np.clip(q_cmd, self.d.qpos - self.cfg.max_lead, self.d.qpos + self.cfg.max_lead)
        self.q_cmd = np.clip(q_cmd, self.q_lo, self.q_hi)
        for _ in range(self.n_sub):                       # the low-level joint PD
            tau = (self.Kp * (self.q_cmd - self.d.qpos) - self.Kd * self.d.qvel
                   + self.d.qfrc_bias)
            self.d.ctrl[:] = np.clip(tau, -self.tau_lim, self.tau_lim)
            mujoco.mj_step(self.m, self.d)
        self._t += 1
        self._update_pose()

        ep, eo = float(np.linalg.norm(self.pos_err)), float(np.linalg.norm(self.ori_err))
        success = is_success(ep, eo)
        reward = float(self.reward_fn(self.pos_err, self.ori_err, a))
        terminated = bool(success and self.cfg.terminate_on_success)
        truncated = self._t >= self.cfg.max_steps
        info = {"pos_err": ep, "ori_err": eo, "is_success": float(success)}
        return self._obs(), reward, terminated, truncated, info


def make_env(reward_fn, targets=None, target_sampler=None, cfg=None, seed=0):
    def _thunk():
        env = ForteReachEnv(reward_fn, targets=targets, target_sampler=target_sampler, cfg=cfg)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _thunk
