"""Where targets live and what counts as reaching one.  (Given.)

The task region is a box in front of the robot, from just above the floor
(z = 0.05 m) up to 0.65 m, with the tool orientation limited to

    |roll| <= 70 deg,  |pitch| <= 50 deg,  |yaw| <= 30 deg      for R = Rz Ry Rx

around the tool pointing straight forward (+x). Low targets are deliberate: near
the floor the elbow and wrist can hit the ground, so collision checking matters.
"""
from __future__ import annotations

import numpy as np

from .robot import rpy_from_R

BOX_CENTER = np.array([0.6, 0.0, 0.35])     # [m]  x forward, y left, z up
BOX_HALF = 0.3                              # [m]  half-width of the cube
RPY_LIMIT = np.radians([70.0, 50.0, 30.0])   # [rad] |roll|, |pitch|, |yaw| limits

POS_TOL = 0.02                               # [m]   success: position error below this
ORI_TOL = 0.20                               # [rad] ...and orientation error below this


def in_task_region(p, R) -> bool:
    """Is the tool pose (p, R) inside the box and the orientation limits?"""
    if np.any(np.abs(np.asarray(p) - BOX_CENTER) > BOX_HALF):
        return False
    return bool(np.all(np.abs(rpy_from_R(R)) <= RPY_LIMIT))


def is_success(pos_err: float, ori_err: float) -> bool:
    return bool(pos_err < POS_TOL and ori_err < ORI_TOL)
