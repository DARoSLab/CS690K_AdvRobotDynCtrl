"""A compact, readable PPO for continuous control.  (Given -- read it.)

Close to CleanRL's `ppo_continuous_action.py`, trimmed to the essentials:

    1. collect `num_steps` transitions from each of `num_envs` environments
    2. compute advantages with Generalized Advantage Estimation (GAE)
    3. for `update_epochs`, shuffle the batch and take minibatch steps on the
       clipped PPO objective + value loss
    4. repeat until `total_timesteps`

Two additions for this homework:
  * `init=` starts the actor from an imitation-learned network (Q4).
  * every `eval_every` updates the greedy policy is evaluated on a FIXED set of
    targets, so learning curves of different runs are directly comparable. The
    policy at each evaluation is also saved as an ordinary checkpoint in a folder
    next to `ckpt_path` (`checkpoints/my_rl_box/step_0409600.pt`, ...), so a whole
    learning curve can be re-tested on other targets later (`load_snapshots`).
"""
from __future__ import annotations

import glob
import os
import shutil
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import EnvConfig, ForteReachEnv, make_env
from .networks import ActorCritic
from .utils import ReturnNormalizer, RunningMeanStd, set_global_seed


@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000
    num_envs: int = 16
    num_steps: int = 128
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    num_minibatches: int = 16
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    target_kl: Optional[float] = 0.03
    hidden_sizes: Tuple[int, ...] = (256, 256)
    logstd_init: float = -1.0       # initial exploration: exp(-1) = 0.37 of a full action
    seed: int = 0
    torch_threads: int = 4
    eval_every: int = 25            # updates between evaluations
    critic_warmup: int = 0          # updates that train ONLY the critic (used with init=)
    eval_n: int = 100               # targets in each evaluation

    @property
    def batch_size(self): return self.num_envs * self.num_steps
    @property
    def minibatch_size(self): return self.batch_size // self.num_minibatches
    @property
    def num_updates(self): return self.total_timesteps // self.batch_size


# ---------------------------------------------------------------------------- #
class Policy:
    """A trained network packaged as `policy(obs) -> action` (greedy: the mean)."""

    def __init__(self, agent: ActorCritic, obs_rms: Optional[RunningMeanStd], obs_clip=10.0):
        self.agent, self.obs_rms, self.obs_clip = agent, obs_rms, obs_clip

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        o = obs[None] if obs.ndim == 1 else obs
        if self.obs_rms is not None:
            o = self.obs_rms.normalize(o, clip=self.obs_clip)
        a = self.agent.act_deterministic(torch.as_tensor(o, dtype=torch.float32)).numpy()
        return a[0] if obs.ndim == 1 else a


def save_checkpoint(path, agent, obs_rms, env_cfg, ppo_cfg, history=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": agent.state_dict(),
                "obs_rms": obs_rms.state_dict() if obs_rms is not None else None,
                "obs_dim": agent.critic[0].in_features,
                "act_dim": agent.actor_logstd.shape[-1],
                "env_cfg": env_cfg, "ppo_cfg": ppo_cfg, "history": history}, path)


def load_checkpoint(path) -> Tuple[Policy, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    agent = ActorCritic(ck["obs_dim"], ck["act_dim"], hidden=tuple(ck["ppo_cfg"].hidden_sizes))
    agent.load_state_dict(ck["model_state"]); agent.eval()
    rms = None
    if ck["obs_rms"] is not None:
        rms = RunningMeanStd(shape=(ck["obs_dim"],)); rms.load_state_dict(ck["obs_rms"])
    return Policy(agent, rms, ck["env_cfg"].obs_clip), ck


def load_snapshots(folder):
    """The policies saved at every evaluation of a training run, from its folder
    (e.g. `checkpoints/my_rl_box`): a list of (step, policy), in order of steps."""
    files = sorted(glob.glob(os.path.join(folder, "step_*.pt")))
    if not files:
        raise FileNotFoundError(f"no step_*.pt files in {folder}: train this run (again) to create them")
    return [(int(os.path.basename(f)[5:-3]), load_checkpoint(f)[0]) for f in files]


def init_from_checkpoint(path, logstd=None) -> dict:
    """An `init` dict for `train` that continues from a saved PPO policy."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = {k[len("actor_mean."):]: v for k, v in ck["model_state"].items() if k.startswith("actor_mean.")}
    rms = RunningMeanStd(shape=(ck["obs_dim"],)); rms.load_state_dict(ck["obs_rms"])
    ls = float(ck["model_state"]["actor_logstd"].mean()) if logstd is None else logstd
    return {"actor_mean": actor, "obs_rms": rms, "logstd": ls}


# ---------------------------------------------------------------------------- #
@torch.no_grad()
def quick_eval(policy: Policy, reward_fn, eval_targets, env_cfg: EnvConfig, n: int) -> dict:
    """Greedy rollouts on the first n targets of a fixed set."""
    P, R = eval_targets
    env = ForteReachEnv(reward_fn, targets=(P, R), cfg=env_cfg)
    succ, pos, ori = [], [], []
    for i in range(min(n, len(P))):
        obs, _ = env.reset(seed=i, options={"target_index": i})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        succ.append(info["is_success"]); pos.append(info["pos_err"]); ori.append(info["ori_err"])
    return {"success": float(np.mean(succ)), "pos_med": float(np.median(pos)),
            "ori_med": float(np.median(ori))}


def train(reward_fn: Callable, targets=None, target_sampler=None, eval_targets=None,
          env_cfg: Optional[EnvConfig] = None, ppo_cfg: Optional[PPOConfig] = None,
          init: Optional[dict] = None, ckpt_path: Optional[str] = None,
          verbose: bool = True) -> Tuple[Policy, dict]:
    """Train a policy with PPO.

    reward_fn       your reward, reward_fn(pos_err, ori_err, action)
    targets         (P, R) arrays: training targets, one drawn per episode   -- or --
    target_sampler  f(rng) -> (p, R): a fresh target every episode
    eval_targets    (P, R): fixed evaluation set for the learning curve
    init            optional {'actor_mean': state_dict, 'obs_rms': RunningMeanStd,
                    'logstd': float} to start from an imitation-learned actor.
                    Two things then change, both to protect what imitation learned:
                    the observation normalization is FROZEN (the cloned network was
                    trained on those exact input statistics), and for the first
                    `critic_warmup` updates only the critic learns. A freshly
                    initialized critic gives meaningless advantages, and policy
                    updates driven by them destroy a good starting policy.
    Returns (policy, history) where history has lists 'step', 'minutes',
    'success', 'pos_med', 'ori_med'.
    """
    env_cfg = env_cfg or EnvConfig()
    cfg = ppo_cfg or PPOConfig()
    set_global_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    import gymnasium as gym

    envs = gym.vector.SyncVectorEnv([
        make_env(reward_fn, targets=targets, target_sampler=target_sampler, cfg=env_cfg,
                 seed=cfg.seed + i) for i in range(cfg.num_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    agent = ActorCritic(obs_dim, act_dim, hidden=cfg.hidden_sizes, logstd_init=cfg.logstd_init)
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    if init is not None:
        agent.actor_mean.load_state_dict(init["actor_mean"])
        if "obs_rms" in init and init["obs_rms"] is not None:
            obs_rms = init["obs_rms"]
        if "logstd" in init:
            agent.actor_logstd.data.fill_(float(init["logstd"]))
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    ret_norm = ReturnNormalizer(cfg.num_envs, cfg.gamma)
    policy = Policy(agent, obs_rms, env_cfg.obs_clip)

    S, N = cfg.num_steps, cfg.num_envs
    obs_buf = torch.zeros((S, N, obs_dim)); act_buf = torch.zeros((S, N, act_dim))
    logp_buf = torch.zeros((S, N)); rew_buf = torch.zeros((S, N))
    done_buf = torch.zeros((S, N)); val_buf = torch.zeros((S, N))

    freeze_rms = init is not None

    def norm_obs(o):
        if not freeze_rms:
            obs_rms.update(o)
        return obs_rms.normalize(o, clip=env_cfg.obs_clip)

    raw, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(norm_obs(raw), dtype=torch.float32)
    next_done = torch.zeros(N)
    hist = {"step": [], "minutes": [], "success": [], "pos_med": [], "ori_med": []}
    t0 = time.time(); step = 0; best = -1.0

    # the policy at every evaluation goes to a folder named after ckpt_path; it is
    # emptied first, so a retrained run never mixes with files of an earlier one
    snap_dir = ckpt_path[:-3] if ckpt_path and ckpt_path.endswith(".pt") else None
    if snap_dir:
        shutil.rmtree(snap_dir, ignore_errors=True)
        os.makedirs(snap_dir)

    def do_eval():
        if eval_targets is None:
            return
        ev = quick_eval(policy, reward_fn, eval_targets, env_cfg, cfg.eval_n)
        hist["step"].append(step); hist["minutes"].append((time.time() - t0) / 60)
        for k in ("success", "pos_med", "ori_med"):
            hist[k].append(ev[k])
        if verbose:
            print(f"  step {step:>9,}  {hist['minutes'][-1]:5.1f} min  success {100*ev['success']:5.1f}%  "
                  f"pos {1000*ev['pos_med']:6.1f} mm  ori {ev['ori_med']:.3f} rad  "
                  f"std {float(torch.exp(agent.actor_logstd.detach()).mean()):.3f}")
        if snap_dir:
            save_checkpoint(os.path.join(snap_dir, f"step_{step:07d}.pt"), agent, obs_rms, env_cfg, cfg)
        nonlocal best
        if ckpt_path and ev["success"] >= best:
            best = ev["success"]
            save_checkpoint(ckpt_path, agent, obs_rms, env_cfg, cfg, hist)

    do_eval()
    for update in range(1, cfg.num_updates + 1):
        if cfg.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / cfg.num_updates) * cfg.learning_rate

        # ----------------- collect a rollout -----------------
        for t in range(S):
            step += N
            obs_buf[t] = next_obs; done_buf[t] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            val_buf[t], act_buf[t], logp_buf[t] = value, action, logp
            raw, reward, term, trunc, _ = envs.step(action.numpy())
            done = np.logical_or(term, trunc)
            r = np.clip(ret_norm(reward, done), -10.0, 10.0)
            nxt = norm_obs(raw)
            # A time-limit cut is not a real end: add the missing tail gamma * V(s_final).
            boot = np.logical_and(trunc, np.logical_not(term))
            if boot.any():
                with torch.no_grad():
                    r = r + cfg.gamma * agent.get_value(torch.as_tensor(nxt, dtype=torch.float32)).numpy() * boot
            rew_buf[t] = torch.as_tensor(r, dtype=torch.float32)
            next_obs = torch.as_tensor(nxt, dtype=torch.float32)
            next_done = torch.as_tensor(done.astype(np.float32))

        # ----------------- GAE -----------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            adv = torch.zeros_like(rew_buf); last = torch.zeros(N)
            for t in reversed(range(S)):
                nonterm = 1.0 - (next_done if t == S - 1 else done_buf[t + 1])
                nextval = next_value if t == S - 1 else val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * nextval * nonterm - val_buf[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                adv[t] = last
            ret = adv + val_buf

        b_obs, b_act = obs_buf.reshape(-1, obs_dim), act_buf.reshape(-1, act_dim)
        b_logp, b_adv = logp_buf.reshape(-1), adv.reshape(-1)
        b_ret, b_val = ret.reshape(-1), val_buf.reshape(-1)
        # rows right after an episode ended are gymnasium's auto-reset dummies: drop them
        valid = np.where(done_buf.reshape(-1).numpy() < 0.5)[0]

        # ----------------- PPO update -----------------
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(valid)
            for s0 in range(0, len(valid), cfg.minibatch_size):
                mb = valid[s0:s0 + cfg.minibatch_size]
                _, newlogp, _, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]; ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                a_mb = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-a_mb * ratio,
                                    -a_mb * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v_clip = b_val[mb] + torch.clamp(newval - b_val[mb], -cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * torch.max((newval - b_ret[mb]) ** 2, (v_clip - b_ret[mb]) ** 2).mean()
                warm = init is not None and update <= cfg.critic_warmup
                loss = (0.0 if warm else pg_loss) + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                if not torch.isfinite(loss):          # never step on a non-finite loss
                    continue
                loss.backward()
                if not torch.isfinite(nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)):
                    optimizer.zero_grad(set_to_none=True); continue
                optimizer.step()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl and not warm:
                break

        if update % cfg.eval_every == 0 or update == cfg.num_updates:
            do_eval()

    envs.close()
    if ckpt_path:
        save_checkpoint(ckpt_path.replace(".pt", "_final.pt"), agent, obs_rms, env_cfg, cfg, hist)
    return policy, hist
