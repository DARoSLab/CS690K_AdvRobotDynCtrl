"""A self-contained Gymnasium environment for the 2D walking robot.

It uses the `mujoco` bindings directly (no `gymnasium.envs.mujoco` dependency)
so every line of the physics/observation/reward loop is visible and hackable.

    env = PlanarWalkerEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    frame = env.render()          # HxWx3 uint8

============================================================================
 HW1 -- PART A : OBSERVATION DESIGN                      [student section]
============================================================================
`_get_obs` below is a reference solution.  For the assignment you decide what
the policy gets to see.  Ask yourself: is absolute x position useful?  Are raw
joint velocities enough?  Would foot-contact flags or the phase of the gait
help?
============================================================================
"""
from __future__ import annotations

import os
from typing import Optional

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("Install dependencies first:  pip install -r requirements.txt") from e

from .config import EnvConfig
from .reward import compute_reward, is_healthy

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "planar_walker.xml")

# Names in data.qpos / data.qvel order (for readable debugging / plotting).
# The MJCF keeps the classic Walker2d joint names; the anatomical names are:
#     thigh_joint = hip,   leg_joint = knee,   foot_joint = ankle
QPOS_NAMES = [
    "rootx", "rootz", "rooty",
    "hip_R", "knee_R", "ankle_R",     # thigh_joint, leg_joint, foot_joint
    "hip_L", "knee_L", "ankle_L",     # thigh_left_joint, leg_left_joint, foot_left_joint
]


class PlanarWalkerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 125}

    def __init__(self, cfg: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(_ASSET)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)

        self.nq = self.model.nq          # 9
        self.nv = self.model.nv          # 9
        self.nu = self.model.nu          # 6

        # --- action space: normalized motor commands in [-1, 1] ---
        self.action_space = spaces.Box(
            low=-self.cfg.ctrl_range, high=self.cfg.ctrl_range,
            shape=(self.nu,), dtype=np.float32,
        )

        # --- observation space: size discovered from a dummy obs ---
        mujoco.mj_resetData(self.model, self.data)
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        self._elapsed_steps = 0
        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer = None

    # ------------------------------------------------------------------ #
    # observation                                                        #
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        # === HW SOLUTION START: env.get_obs ===
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        if self.cfg.exclude_current_positions:
            qpos = qpos[1:]              # drop absolute x -> policy is translation invariant
        qvel = np.clip(qvel, -10.0, 10.0)
        return np.concatenate([qpos, qvel]).astype(np.float32)
        # === HW SOLUTION END: env.get_obs ===

    # ------------------------------------------------------------------ #
    # gym API                                                            #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        n = self.cfg.reset_noise_scale
        self.data.qpos[:] += self.np_random.uniform(-n, n, size=self.nq)
        self.data.qvel[:] += self.np_random.uniform(-n, n, size=self.nv)
        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64),
                         -self.cfg.ctrl_range, self.cfg.ctrl_range)

        qpos_before = self.data.qpos.copy()
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.cfg.frame_skip)
        qpos_after = self.data.qpos.copy()
        qvel_after = self.data.qvel.copy()

        reward, rinfo = compute_reward(
            qpos_before, qpos_after, qvel_after, action,
            self.cfg.control_dt, self.cfg.reward,
        )

        terminated = not is_healthy(qpos_after, self.cfg.reward)
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps

        obs = self._get_obs()
        info = dict(rinfo)
        info["is_healthy"] = not terminated
        info["x_position"] = float(qpos_after[0])

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # rendering                                                          #
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track")
            return self._renderer.render()
        if self.render_mode == "human":
            from mujoco import viewer as _mj_viewer

            if self._viewer is None:
                self._viewer = _mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def make_env(cfg: Optional[EnvConfig] = None, seed: int = 0, render_mode=None):
    """Thunk for gymnasium.vector.SyncVectorEnv."""
    def _thunk():
        env = PlanarWalkerEnv(cfg=cfg, render_mode=render_mode)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _thunk
