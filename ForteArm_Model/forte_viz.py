"""Shared visualisation helpers for the ForteArm homeworks (HW4, HW3, HW5).

Everything here is *given* — you never have to modify it. It exists so that the
numbers you compute (a joint configuration, an IK iteration history, a tracked
trajectory) can actually be looked at.

    import sys; sys.path.append('..')
    from forte_viz import ForteViz, show_video, filmstrip

    viz = ForteViz()
    frames = viz.render_configs(q_history, target=(p_des, R_des))
    show_video(frames, fps=10)          # inline video in the notebook
    filmstrip(frames, n=6)              # or a single strip of stills

-------------------------------------------------------------------------------
Rendering notes
-------------------------------------------------------------------------------
* macOS / Windows: works out of the box.
* Linux without a display: MuJoCo needs an offscreen GL context. Either
  `pip install "mujoco[egl]"` or run with `MUJOCO_GL=egl` set.
* Nothing here is needed to *solve* any homework question -- if rendering will
  not cooperate on your machine, the numerical self-tests still tell you whether
  your implementation is right.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Iterable, Optional, Sequence, Tuple

import mujoco
import numpy as np

_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "./", "forte_scene.xml")

Pose = Tuple[np.ndarray, np.ndarray]     # (position 3, rotation 3x3)


class ForteViz:
    """Offscreen renderer for the ForteArm, with a target-pose marker.

    Width/height should be multiples of 16 so ffmpeg does not silently resize.
    Any size works -- the offscreen cap is raised to match. The default is sized
    so that a frame is still legible after a notebook scales it down to the cell
    width; drop it only if render time or notebook size actually becomes a
    problem.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL,
                 width: int = 960, height: int = 704):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        # MuJoCo caps offscreen rendering at the model's <global offwidth/offheight>
        # (the scene ships 1280x960). Raise it to whatever was asked for, so a
        # bigger render never fails with an opaque framebuffer error.
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(width))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(height))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_tip")
        self.mocap = self.model.body("target").mocapid[0]
        self._hide_target()

    # ------------------------------------------------------------------ #
    def _hide_target(self):
        """Park the marker far away when there is no target to show."""
        self.data.mocap_pos[self.mocap] = np.array([0.0, 0.0, -5.0])

    def set_target(self, pose: Optional[Pose]):
        if pose is None:
            self._hide_target(); return
        p, R = pose
        self.data.mocap_pos[self.mocap] = np.asarray(p, float)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(R, float).reshape(-1))
        self.data.mocap_quat[self.mocap] = quat

    def tool_pose(self) -> Pose:
        return (self.data.site_xpos[self.ee].copy(),
                self.data.site_xmat[self.ee].reshape(3, 3).copy())

    # ------------------------------------------------------------------ #
    # A FIXED camera by default. The arm's reach is ~1.18 m, so this framing
    # always contains the whole reachable workspace -- which means two renders
    # of two different configurations are directly comparable. (Auto-framing
    # that chases the arm looks nicer per-frame but makes the scale jump around,
    # so you cannot tell whether the arm moved or the camera did.)
    LOOKAT = np.array([0.05, 0.0, 0.22])
    DISTANCE = 2.85
    AZIMUTH = 140.0
    ELEVATION = -15.0

    def camera(self, zoom: float = 1.0, azimuth: Optional[float] = None,
               elevation: Optional[float] = None, lookat=None):
        """The standard workspace camera. `zoom > 1` moves closer."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = np.asarray(self.LOOKAT if lookat is None else lookat, float)
        cam.distance = self.DISTANCE / max(zoom, 1e-6)
        cam.azimuth = self.AZIMUTH if azimuth is None else azimuth
        cam.elevation = self.ELEVATION if elevation is None else elevation
        return cam

    def frame(self, q: np.ndarray, target: Optional[Pose] = None,
              cam: Optional[object] = None, zoom: float = 1.0) -> np.ndarray:
        """Render one joint configuration to an RGB array."""
        self.data.qpos[:] = np.asarray(q, float).ravel()[: self.model.nq]
        self.set_target(target)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=cam or self.camera(zoom))
        return self.renderer.render()

    def render_configs(self, qs: Iterable[np.ndarray], target: Optional[Pose] = None,
                       targets: Optional[Sequence[Optional[Pose]]] = None,
                       hold_last: int = 0, zoom: float = 1.0):
        """Render a sequence of configurations.

        target    : one pose held for the whole sequence
        targets   : or a per-frame list of poses (for multi-episode rollouts)
        hold_last : repeat the final frame this many times (a beat at the end)
        """
        qs = [np.asarray(q, float).ravel() for q in qs]
        cam = self.camera(zoom)
        out = []
        for i, q in enumerate(qs):
            tgt = targets[i] if targets is not None else target
            out.append(self.frame(q, tgt, cam=cam))
        out += [out[-1]] * hold_last
        return out

    def close(self):
        self.renderer.close()


# ---------------------------------------------------------------------------- #
def show_video(frames, fps: int = 15, path: str = "_viz.mp4", width: int = 720):
    """Write frames to mp4 and return an inline notebook video.

    `width` is the on-screen width of the embedded player, not the encoded
    resolution -- that comes from the frames, i.e. from ForteViz's width/height.
    """
    import imageio
    from IPython.display import Video
    imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)
    return Video(path, embed=True, width=width)


def filmstrip(frames, n: int = 6, path: Optional[str] = None,
              cols: Optional[int] = None):
    """Lay n evenly spaced frames out as an image. Returns the image array.

    cols=None gives one long row (the original behaviour). Any other value wraps
    into a grid of that many columns, which is usually what you want for more
    than three or four frames: a notebook scales the whole image down to the cell
    width, so a single row of six frames leaves each one about a thumbnail wide.
    """
    frames = list(frames)
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    picked = [frames[i] for i in idx]
    if cols is None or cols >= len(picked):
        grid = np.concatenate(picked, axis=1)
    else:
        h, w = picked[0].shape[:2]
        blank = np.zeros_like(picked[0])
        rows = []
        for r in range(0, len(picked), cols):
            row = picked[r:r + cols]
            row = row + [blank] * (cols - len(row))       # pad the last row
            rows.append(np.concatenate(row, axis=1))
        grid = np.concatenate(rows, axis=0)
    if path:
        import imageio
        imageio.imwrite(path, grid)
    return grid


def show_filmstrip(frames, n: int = 6, title: str = "",
                   cols: Optional[int] = None, frame_in: float = 4.2,
                   dpi: int = 140):
    """Plot a filmstrip with matplotlib (works even where video embedding does not).

    frame_in : on-screen width allotted to ONE frame, in inches. The figure is
               sized from the image's own aspect ratio, so frames are never
               squashed and never silently downsampled to a 100-dpi default.
    """
    import matplotlib.pyplot as plt
    grid = filmstrip(frames, n, cols=cols)
    n_cols = min(n, len(frames)) if cols is None else min(cols, n, len(frames))
    w_in = frame_in * n_cols
    h_in = w_in * grid.shape[0] / grid.shape[1]
    fig = plt.figure(figsize=(w_in, h_in + (0.45 if title else 0.0)), dpi=dpi)
    plt.imshow(grid, interpolation='antialiased'); plt.axis('off')
    if title:
        plt.title(title, fontsize=11)
    plt.tight_layout(); plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------- #
def play_interactive(viz: "ForteViz", episodes, fps: int = 20, hold: float = 0.8,
                     loop: bool = True):
    """Play episodes in MuJoCo's INTERACTIVE viewer instead of recording a video.

    The window is a real MuJoCo viewer: drag to orbit, scroll to zoom, right-drag
    to pan, double-click to centre on a body. This function only drives the
    robot's configuration and the target marker -- it never touches the camera,
    so whatever view you set up persists as the robot moves.

    episodes : iterable of (qs, target_pose) -- qs is a joint trajectory,
               target_pose is (p, R) or None.
    fps      : playback pacing. Lower it to study a fast motion.
    hold     : seconds to dwell on the final pose of each episode.
    loop     : replay the whole sequence until the window is closed (default).
               Without this a short rollout is over in a second and the window
               vanishes before you have had a chance to move the camera.

    macOS: MuJoCo requires the interactive viewer to own the main thread, so this
    must be run with `mjpython`, not `python`. The error below says so if you
    forget. Everything else in this file works under plain `python`.
    """
    # NB: alias the import. A bare `import mujoco.viewer` inside a function binds
    # the name `mujoco` locally and shadows the module-level import, which throws
    # UnboundLocalError on the next `mujoco.` call in this scope.
    from mujoco import viewer as _mj_viewer

    try:
        ctx = _mj_viewer.launch_passive(viz.model, viz.data)
    except Exception as e:                                   # noqa: BLE001
        if sys.platform == "darwin":
            raise SystemExit(
                "\nMuJoCo's interactive viewer must own the main thread on macOS,\n"
                "so it has to be launched with mjpython. Re-run this exact command as:\n\n"
                "    mjpython " + " ".join(sys.argv) + "\n\n"
                "(mjpython ships with the mujoco pip package.)\n"
                f"underlying error: {e}") from None
        raise

    dt = 1.0 / max(int(fps), 1)
    print("  [viewer] drag to orbit, scroll to zoom, right-drag to pan."
          + ("  Replaying until you close the window." if loop else ""))
    with ctx as v:
        running = True
        while running:
            for qs, target in episodes:
                if not v.is_running():
                    running = False
                    break
                viz.set_target(target)
                for q in qs:
                    if not v.is_running():
                        running = False
                        break
                    viz.data.qpos[:] = np.asarray(q, float).ravel()[: viz.model.nq]
                    mujoco.mj_forward(viz.model, viz.data)
                    v.sync()
                    time.sleep(dt)
                t0 = time.time()
                while v.is_running() and time.time() - t0 < hold:
                    v.sync()
                    time.sleep(dt)
            if not loop:
                running = False
