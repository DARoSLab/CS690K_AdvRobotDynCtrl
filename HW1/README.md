# HW1 — Train a 2D Walking Robot with Reinforcement Learning

**Advanced Robot Dynamics and Control** · Planar Walking Robot

In this assignment you train a control policy that makes a planar (2D) bipedal
robot walk forward, using **Proximal Policy Optimization (PPO)**. Everything runs
on a **laptop CPU** — no GPU, no cloud, any OS.

You will implement three things:

| Part | File | What you build |
|------|------|----------------|
| **A. Observation** | `planar_walker/env.py` → `_get_obs` | what the policy senses |
| **B. Reward** | `planar_walker/reward.py` → `compute_reward`, `is_healthy` | what "good walking" means |
| **C. Policy/Value network** | `planar_walker/networks.py` → `ActorCritic` | the function approximator |

The PPO training loop itself (`planar_walker/ppo.py`) is **given and fully
working** — read it, don't rewrite it. Later in the course we will re-derive the
same walking behavior with model-based trajectory optimization and PD control,
and compare.

---

## 1. The robot

A torso plus two legs, confined to the sagittal (x–z) plane.

```
 generalized coordinates  qpos (9):   [ rootx, rootz, rooty,
                                         hip_R, knee_R, ankle_R,
                                         hip_L, knee_L, ankle_L ]
 floating base            3 DoF:       rootx (forward), rootz (height), rooty (pitch)
 actuated joints          6:           hip / knee / ankle, per leg
 action                   a ∈ [-1, 1]^6   normalized motor torques (gear = 100)
 control rate             125 Hz (frame_skip = 4, sim dt = 2 ms)
```

Model file: `planar_walker/assets/planar_walker.xml` — the standard MuJoCo
`Walker2d-v5` model (joint names `thigh/leg/foot` = `hip/knee/ankle`). You may
tune the XML, but you do not need to.

---

## 2. Setup

```bash
cd Homework/PlanarWalking
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip install torch` tries to download a huge CUDA build, force the CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Check your install** (loads MuJoCo, steps the env with random torques):

```bash
python play_random.py --steps 200
python play_random.py --video random.mp4      # writes a short clip
```

Rendering notes:
- macOS / Windows: works out of the box.
- Linux headless (no display): offscreen rendering needs an EGL/OSMesa GL
  context. `pip install "mujoco[egl]"` or run with `MUJOCO_GL=egl`. Training
  needs **no** rendering, only `evaluate.py --video` does.

---

## 3. Train

```bash
python train.py                              # ~1.2M steps, ~4–6 min on a modern laptop CPU
python train.py --total-timesteps 300000     # quick sanity run
python train.py --seed 1 --target-speed 1.2  # try overrides
```

A clean forward gait appears around 0.3–0.5M steps; `checkpoints/policy.pt`
always holds the best evaluation so far, so you can stop early with Ctrl-C.

Console output per update:

```
upd  120  step   983,040   5100 sps  epRet    780.4  KL 0.018  std 0.42
  [eval] step=   983,040  return=  1150.3 +/-  40.2  dist= 11.84 m  speed= 1.42 m/s
```

Checkpoints are written to `checkpoints/`:
- `policy.pt` — best eval return so far (use this one)
- `policy_final.pt` — end of training

A **reference solution checkpoint** is in `checkpoints/reference_policy.pt` so
you can see the target behavior before your own run finishes.

## 4. Evaluate & visualize

```bash
python evaluate.py --checkpoint checkpoints/policy.pt --episodes 10
python evaluate.py --checkpoint checkpoints/policy.pt --video walk.mp4
python evaluate.py --checkpoint checkpoints/policy.pt --viewer     # interactive window
(or mjpython on macOS)
```

`notebook.ipynb` runs the whole flow (train → curves → video) inline.

---

## 5. What to hand in

A **PDF report** (via the notebook or your own writeup) plus your modified code:

1. **Observation (Part A).** What did you include and why? Show one ablation
   (e.g. train with vs. without base pitch, or with vs. without velocities) and
   its learning curves.
2. **Reward (Part B).** Give your final reward as an equation. Explain each term.
   Show at least one *failed* reward you tried and what pathological gait it
   produced (screenshots/short description).
3. **Network (Part C).** Report your architecture. Sweep one hyperparameter
   (width, depth, or state-dependent vs. state-independent log-std) and plot the
   effect on return.
4. **Result.** Learning curve (return vs. steps), evaluation table
   (return / distance / speed / episode length over ≥10 seeds-or-episodes), and a
   video/GIF of the final gait.
5. **Reflection (½ page).** Where did PPO get stuck? What reward term or
   observation fixed it? What do you expect a model-based controller to do
   differently on this same robot?

**Grading (100 pts):** A 20 · B 30 · C 20 · Result 20 · Reflection 10.
Bonus (10): reach ≥ 1.3 m/s average speed with episode length ≥ 8 s, or make the
robot walk *backward* by changing only the reward.

---

## 6. File map

```
planar_walker/
  config.py      all hyperparameters (dataclasses) — start here
  env.py         PlanarWalkerEnv (Gymnasium API, pure `mujoco`)     [PART A]
  reward.py      compute_reward / is_healthy                         [PART B]
  networks.py    ActorCritic MLP + Gaussian policy                   [PART C]
  ppo.py         PPO trainer, GAE, evaluation, checkpoint IO   (given, read it)
  utils.py       seeding, running mean/std, return normalizer
  assets/planar_walker.xml
train.py         CLI: run training
evaluate.py      CLI: roll out a checkpoint, save video, or open viewer
play_random.py   CLI: env sanity check
notebook.ipynb   guided end-to-end version
```

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `mujoco` import error | `pip install -r requirements.txt`; on Linux see §2 rendering notes |
| Training uses all CPU cores / laptop fans spin up | lower `--torch-threads` and `--num-envs` |
| Return climbs then collapses | PPO KL too large — lower `--learning-rate`, or it's a reward exploit |
| Robot "walks" by diving forward | your forward-reward term has no speed cap / no posture term |
| `evaluate.py --video` fails | `pip install imageio-ffmpeg`, or write a `.gif` instead of `.mp4` |
| `OMP: Error #15 ... libomp.dylib already initialized` (macOS/conda) | use a fresh `python -m venv` (not conda base), or as a last resort `export KMP_DUPLICATE_LIB_OK=TRUE` |
