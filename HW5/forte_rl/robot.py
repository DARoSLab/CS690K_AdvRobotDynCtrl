"""The ForteArm, its kinematics, and a collision checker.  (Given -- read it.)

Everything here is plumbing you already know from the previous homeworks:
forward kinematics, the pose error ordered [omega; v], and the rotation helpers.
The one new piece is `CollisionChecker`.

Why a separate collision checker?  The shared model `Forte.xml` has collisions
switched OFF on purpose: contact forces would disturb the dynamics in the
operational-space-control homework. Here we need to *detect* collisions, not
simulate them, so the checker compiles its own private copy of the model with
collisions switched on, poses it, and lists the contacts. It never steps the
physics.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

MODEL_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "ForteArm_Model", "forte_scene.xml"))

JOINT_NAMES = ["shoulder_yaw", "shoulder_pitch", "shoulder_roll", "elbow_pitch",
               "lower_arm_roll", "wrist_pitch", "wrist_roll"]


# --------------------------------------------------------------------------- #
#  rotation helpers                                                            #
# --------------------------------------------------------------------------- #
def log_SO3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis*angle 3-vector."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    if c > 1.0 - 1e-12:
        S = (R - R.T) / 2.0
        return np.array([S[2, 1], S[0, 2], S[1, 0]])
    if c < -1.0 + 1e-12:
        A = R + np.eye(3)
        k = int(np.argmax(np.linalg.norm(A, axis=0)))
        return np.pi * A[:, k] / np.linalg.norm(A[:, k])
    th = np.arccos(c)
    S = (R - R.T) / (2.0 * np.sin(th))
    return th * np.array([S[2, 1], S[0, 2], S[1, 0]])


def pose_error(p, R, p_des, R_des) -> np.ndarray:
    """6-D pose error [omega; v]: rotation error first, then position error."""
    return np.concatenate([log_SO3(R_des @ R.T), p_des - p])


def rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened: a continuous, unique encoding of a
    rotation that a network can read (Euler angles jump; quaternions double-cover)."""
    return np.asarray(R)[..., :, :2].swapaxes(-1, -2).reshape(*np.shape(R)[:-2], 6)


def rpy_from_R(R: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw [rad] for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return np.array([np.arctan2(R[2, 1], R[2, 2]),
                     np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)),
                     np.arctan2(R[1, 0], R[0, 0])])


def R_from_rpy(rpy) -> np.ndarray:
    """Rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


# --------------------------------------------------------------------------- #
#  the arm                                                                     #
# --------------------------------------------------------------------------- #
class Arm:
    """Model, joint limits, forward kinematics and the tool Jacobian."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.nq = self.m.nq
        self.ee = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "ee_tip")
        self.q_lo = self.m.jnt_range[:, 0].copy()
        self.q_hi = self.m.jnt_range[:, 1].copy()
        self.q_home = self.m.key_qpos[1].copy()          # `ready_pose`
        self.tau_lim = self.m.actuator_ctrlrange[:, 1].copy()

    def fk(self, q):
        """Tool position (3,) and orientation (3x3) for joint angles q."""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.ee].copy(), self.d.site_xmat[self.ee].reshape(3, 3).copy()

    def jacobian(self, q):
        """6 x 7 world-frame tool Jacobian, ordered [omega; v]."""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        Jp = np.zeros((3, self.nq)); Jr = np.zeros((3, self.nq))
        mujoco.mj_jacSite(self.m, self.d, Jp, Jr, self.ee)
        return np.vstack([Jr, Jp])


# --------------------------------------------------------------------------- #
#  collision checking                                                          #
# --------------------------------------------------------------------------- #
class CollisionChecker:
    """Does the arm, in configuration q, touch the floor or itself?

        cc = CollisionChecker()
        hit = cc.check(q)       # {'ground': bool, 'self': bool, 'pairs': [...]}

    Collision shapes are the convex hulls of the link meshes, so the check is
    slightly conservative: it can report contact a few millimetres before the real
    parts would touch. One pair is ignored: `shoulder_yaw_link` and
    `upper_arm_link`, whose hulls overlap in EVERY configuration (measured over
    3,000 random poses), because the shoulder pitch link sits between them and the
    hulls do not see the gap. MuJoCo already ignores parent-child pairs.
    """

    IGNORE = (("shoulder_yaw_link_geom", "upper_arm_link_geom"),)

    def __init__(self, model_path: str = MODEL_PATH):
        spec = mujoco.MjSpec.from_file(model_path)
        for g in spec.geoms:
            if g.name == "base_link_geom":       # visual base: overlaps the shoulder by design
                continue
            g.contype, g.conaffinity = 1, 1
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        gid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.floor = gid("floor")
        self.ignore = {tuple(sorted((gid(a), gid(b)))) for a, b in self.IGNORE}
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i)
                      for i in range(self.m.ngeom)]

    def check(self, q) -> dict:
        self.d.qpos[:] = q
        mujoco.mj_fwdPosition(self.m, self.d)     # link poses and contact detection (no dynamics needed)
        ground = self_hit = False
        pairs = []
        for g1, g2 in self.d.contact.geom[:self.d.ncon]:
            pr = (int(min(g1, g2)), int(max(g1, g2)))
            if pr in self.ignore:
                continue
            if (self.names[pr[0]], self.names[pr[1]]) not in pairs:      # one entry per touching pair
                pairs.append((self.names[pr[0]], self.names[pr[1]]))
            if self.floor in pr:
                ground = True
            else:
                self_hit = True
        return {"ground": ground, "self": self_hit, "pairs": pairs}

    def in_collision(self, q) -> bool:
        r = self.check(q)
        return r["ground"] or r["self"]
