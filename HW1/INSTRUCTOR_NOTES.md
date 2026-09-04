# HW1 — Instructor Notes (do not distribute)

## What this folder is

A **complete, working reference solution** for the planar-walking RL homework,
plus tooling (`make_student_version.py`) to strip selected parts and produce the
student handout. Nothing here is redacted yet — you decide what to open.

## Shipped reference artifacts

- `checkpoints/reference_policy.pt` — trained seed-0 policy (return ≈ 1,960).
- `checkpoints/reference_train_log.txt` — the full training log it came from.
- `reference_gait.mp4`, `training_curve.png` — regenerate with the commands at
  the bottom of this file.

The reference gait is a fast forward-leaning stride, not an upright walk — it's
what plain PPO + a minimal reward converges to. That's deliberate: "make it walk
more upright by adding a posture term" is exactly the kind of thing students
explore in Part B and revisit when we do model-based control.

## The five redactable blocks

| block id | file | difficulty | what it teaches |
|---|---|---|---|
| `env.get_obs` | `env.py` | easy–med | observation design, translation invariance, what a policy needs to sense |
| `reward.is_healthy` | `reward.py` | easy | termination vs. reward; the "healthy set" idea (ties to region-of-attraction later) |
| `reward.compute_reward` | `reward.py` | **med–hard** | reward shaping, reward hacking, speed cap vs. linear, control cost trade-off |
| `networks.build_mlp` | `networks.py` | easy | MLP plumbing, orthogonal init |
| `networks.actor_critic_init` | `networks.py` | med | diagonal Gaussian policy, state-independent log-std, separate vs. shared trunk |

`ppo.py` (the algorithm) is **never redacted** — students read it, and it is the
bridge to the model-based half of the course (GAE ↔ value functions, the PPO
trust region ↔ line search in optimization).

### Recommended release configurations

- **First offering / short timeline:** `--preset reward-only`. Students get a
  working obs + network and only design the reward. This is the highest-value,
  lowest-frustration option and still surfaces reward hacking.
- **Standard:** `--preset reward+net`. Adds the actor-critic. ~1 week of work.
- **Full / grad-heavy:** `--preset full`. Also redacts `_get_obs`. Do this only
  if you have lecture time for observation/normalization discussion first.

```bash
python make_student_version.py --preset reward+net --out ../PlanarWalking_release
```

Then drop `checkpoints/reference_policy.pt` into the release so students can see
the target gait without waiting for their first successful run.

## Expected results (reference solution, laptop CPU)

- `train.py` defaults: 8 envs × 512 steps, 1.2 M steps.
- Throughput on an M-series / recent x86 laptop: ~6,500 env-steps/s
  → **~3–5 minutes** for the default run.
- A clean forward gait emerges around 0.3–0.5 M steps; the reference seed 0
  reaches eval-return ≈ 1,960 (near the ~2,000 ceiling) by ~0.4 M.
- Final greedy-eval (reference `policy.pt`): return ≈ 1,960, distance ≈ 14 m over
  an 8 s episode, average speed ≈ 1.7 m/s.
- Walker2d PPO is somewhat seed-sensitive; ~1 in 4 seeds learns a hopping/limping
  gait instead. Tell students to run 2–3 seeds and grade on the best. `policy.pt`
  keeps the best eval checkpoint, so training past the peak is harmless (the
  stochastic-policy return drifts down ~3% and distance more — see
  `training_curve.png`).

## Known failure modes students will hit (good discussion material)

| symptom | cause | the lesson |
|---|---|---|
| robot lunges forward and faceplants, return plateaus ~100 | forward reward with no speed cap and/or no ctrl cost; falls before `is_healthy` fires | greedy reward → ballistic "gait"; need a bounded objective |
| stands still forever, 0 m distance | alive bonus ≫ forward term | reward has to *pay* for the risky behavior you want |
| violent shaking, huge torques | ctrl cost = 0 | regularization; ties to effort/energy cost in optimal control |
| great training return, terrible eval | reward exploits normalization / early-term quirks, or overfits stochastic policy | always evaluate the greedy policy separately |
| learns to walk backward | forward term rewards `|velocity|` or wrong sign | sign conventions matter |

## Grading

Rubric is in `README.md` §5. Suggested weighting inside "Result" (20 pts): a gait
that stays up ≥ 6 s and travels ≥ 5 m gets full marks; partial credit for a
policy that clearly learned *something* (return well above the ~100 lunge floor).

The **reflection** (10 pts) is where they connect to the analytical half — look
for a concrete, specific claim about what a model-based controller would do
differently (e.g. "MPC would plan the foot placement instead of discovering it,"
"a PD controller needs a reference trajectory we don't have yet").

## Regenerating the reference checkpoint

```bash
python train.py --total-timesteps 2000000 --seed 0 --checkpoint checkpoints/reference_policy.pt
python evaluate.py --checkpoint checkpoints/reference_policy.pt --video reference_gait.mp4 --episodes 20
```

If seed 0 gives a poor gait after the model/rig changes, sweep `--seed 0..5` and
keep the best `policy.pt` (renamed).

## Portability checklist (the "any laptop, no NVIDIA GPU" requirement)

- `mujoco` wheels: prebuilt for macOS (x86 + arm64), Windows, manylinux. CPU physics.
- `torch`: default PyPI wheel is CPU-only; `--index-url .../whl/cpu` documented in `requirements.txt`.
- Training does **no rendering**. Only `evaluate.py --video` / `play_random.py --video`
  need a GL context. Linux headless: `MUJOCO_GL=egl` (documented in README §2).
- No `AsyncVectorEnv` (multiprocessing pickling issues on Windows/macOS-spawn) —
  we use `SyncVectorEnv`.
- `torch.set_num_threads(cfg.torch_threads)` keeps a laptop usable during training.
- **CPU-only by design.** `train()` hardcodes `device = "cpu"`. A GPU does not
  help: the rollout is CPU-bound (MuJoCo) and the 2x64 net is far too small to
  amortise host<->device copies (measured ~8x *slower* on Apple MPS). If you ever
  want GPU-parallel sim for the KAIST hands-on version, that means MJX / MuJoCo
  Warp + a batched-env rewrite — a separate optional track, not this pipeline.
