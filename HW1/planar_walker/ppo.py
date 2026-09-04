"""A compact, readable PPO for continuous control (CPU-friendly).

This is deliberately close to CleanRL's `ppo_continuous_action.py` so you can
cross-reference it with the literature, but it is trimmed to the essentials and
wired to our `PlanarWalkerEnv`.

The training loop is intentionally NOT a student section -- you are meant to be
able to read it end to end and understand every step:

    1. collect a rollout of `num_steps` transitions from `num_envs` envs
    2. compute advantages with Generalized Advantage Estimation (GAE)
    3. for `update_epochs`, shuffle the batch and take minibatch SGD steps on
       the clipped PPO objective + value loss + entropy bonus
    4. repeat
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import EnvConfig, PPOConfig
from .env import PlanarWalkerEnv, make_env
from .networks import ActorCritic
from .utils import ReturnNormalizer, RunningMeanStd, Stopwatch, set_global_seed


# ---------------------------------------------------------------------------- #
# evaluation                                                                   #
# ---------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_policy(
    agent: ActorCritic,
    obs_rms: Optional[RunningMeanStd],
    env_cfg: EnvConfig,
    n_episodes: int = 5,
    seed: int = 10_000,
    obs_clip: float = 10.0,
    device: str = "cpu",
) -> dict:
    """Roll out the greedy (mean) policy and report episode return / length / distance."""
    env = PlanarWalkerEnv(cfg=env_cfg)
    returns, lengths, distances, speeds = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            o = obs_rms.normalize(obs[None], clip=obs_clip) if obs_rms is not None else obs[None]
            a = agent.act_deterministic(torch.as_tensor(o, dtype=torch.float32, device=device))
            obs, r, term, trunc, info = env.step(a.cpu().numpy()[0])
            ep_ret += r
            ep_len += 1
            done = term or trunc
        returns.append(ep_ret)
        lengths.append(ep_len)
        distances.append(info["x_position"])
        speeds.append(info["x_position"] / (ep_len * env_cfg.control_dt))
    env.close()
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/length_mean": float(np.mean(lengths)),
        "eval/distance_mean": float(np.mean(distances)),
        "eval/speed_mean": float(np.mean(speeds)),
    }


# ---------------------------------------------------------------------------- #
# checkpoint IO                                                                #
# ---------------------------------------------------------------------------- #
def save_checkpoint(path: str, agent: ActorCritic, obs_rms, env_cfg: EnvConfig, ppo_cfg: PPOConfig):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": agent.state_dict(),
            "obs_rms": obs_rms.state_dict() if obs_rms is not None else None,
            "env_cfg": env_cfg,
            "ppo_cfg": ppo_cfg,
            "hidden_sizes": ppo_cfg.hidden_sizes,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    env_cfg: EnvConfig = ckpt["env_cfg"]
    probe = PlanarWalkerEnv(cfg=env_cfg)
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    probe.close()

    agent = ActorCritic(obs_dim, act_dim, hidden=tuple(ckpt["hidden_sizes"])).to(device)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()

    obs_rms = None
    if ckpt["obs_rms"] is not None:
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        obs_rms.load_state_dict(ckpt["obs_rms"])
    return agent, obs_rms, env_cfg, ckpt.get("ppo_cfg")


# ---------------------------------------------------------------------------- #
# training                                                                     #
# ---------------------------------------------------------------------------- #
def train(
    ppo_cfg: PPOConfig,
    env_cfg: EnvConfig,
    ckpt_path: str = "checkpoints/policy.pt",
    log_fn: Optional[Callable[[dict], None]] = None,
) -> str:
    set_global_seed(ppo_cfg.seed)
    torch.set_num_threads(ppo_cfg.torch_threads)
    device = "cpu"  # MuJoCo runs on CPU and the networks are tiny -- a GPU is no faster here

    # --- vectorized envs (each env has an independent seed) ---
    import gymnasium as gym

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_cfg, seed=ppo_cfg.seed + i) for i in range(ppo_cfg.num_envs)]
    )
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    agent = ActorCritic(obs_dim, act_dim, hidden=ppo_cfg.hidden_sizes).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(obs_dim,)) if ppo_cfg.normalize_obs else None
    ret_norm = ReturnNormalizer(ppo_cfg.num_envs, ppo_cfg.gamma) if ppo_cfg.normalize_reward else None

    # --- rollout storage ---
    S, N = ppo_cfg.num_steps, ppo_cfg.num_envs
    obs_buf = torch.zeros((S, N, obs_dim))
    act_buf = torch.zeros((S, N, act_dim))
    logp_buf = torch.zeros((S, N))
    rew_buf = torch.zeros((S, N))
    done_buf = torch.zeros((S, N))
    val_buf = torch.zeros((S, N))

    def norm_obs(o: np.ndarray) -> np.ndarray:
        if obs_rms is None:
            return o.astype(np.float32)
        obs_rms.update(o)
        return obs_rms.normalize(o, clip=env_cfg.obs_clip)

    next_obs_raw, _ = envs.reset(seed=ppo_cfg.seed)
    next_obs = torch.as_tensor(norm_obs(next_obs_raw), dtype=torch.float32)
    next_done = torch.zeros(N)

    watch = Stopwatch()
    global_step = 0
    ep_returns: list[float] = []          # rolling window of finished-episode returns
    ep_return_running = np.zeros(N)

    best_eval = -1e9

    for update in range(1, ppo_cfg.num_updates + 1):
        if ppo_cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / ppo_cfg.num_updates
            optimizer.param_groups[0]["lr"] = frac * ppo_cfg.learning_rate

        # ----------------- collect rollout -----------------
        for t in range(S):
            global_step += N
            obs_buf[t] = next_obs
            done_buf[t] = next_done

            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            val_buf[t] = value
            act_buf[t] = action
            logp_buf[t] = logp

            raw_next_obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)

            ep_return_running += reward
            for i in range(N):
                if done[i]:
                    ep_returns.append(float(ep_return_running[i]))
                    ep_return_running[i] = 0.0
            ep_returns = ep_returns[-100:]

            r = ret_norm(reward, done) if ret_norm is not None else reward
            r = np.clip(r, -10.0, 10.0)
            rew_buf[t] = torch.as_tensor(r, dtype=torch.float32)
            next_obs = torch.as_tensor(norm_obs(raw_next_obs), dtype=torch.float32)
            next_done = torch.as_tensor(done.astype(np.float32))

        # ----------------- GAE -----------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = torch.zeros(N)
            # NOTE: we treat time-limit truncations like terminations (no value
            # bootstrap from the final state). With an 8 s horizon and gamma 0.99
            # the bias is small; handling it exactly is a nice optional exercise.
            for t in reversed(range(S)):
                nonterminal = 1.0 - (next_done if t == S - 1 else done_buf[t + 1])
                nextval = next_value if t == S - 1 else val_buf[t + 1]
                delta = rew_buf[t] + ppo_cfg.gamma * nextval * nonterminal - val_buf[t]
                lastgaelam = delta + ppo_cfg.gamma * ppo_cfg.gae_lambda * nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + val_buf

        # ----------------- flatten batch -----------------
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        # gymnasium's vector env uses "next-step autoreset": the step right after
        # an episode ends is a dummy (it returns the reset obs, reward 0, and
        # ignores the action). Those transitions are not real -- drop them from
        # the update. `done_buf[t]` is exactly the "this row is a reset dummy" flag.
        valid = np.where(done_buf.reshape(-1).numpy() < 0.5)[0]

        # ----------------- PPO update -----------------
        clipfracs = []
        approx_kl = torch.tensor(0.0)
        for epoch in range(ppo_cfg.update_epochs):
            np.random.shuffle(valid)
            for start in range(0, len(valid), ppo_cfg.minibatch_size):
                mb = valid[start:start + ppo_cfg.minibatch_size]

                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > ppo_cfg.clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - ppo_cfg.clip_coef, 1 + ppo_cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                if ppo_cfg.clip_vloss:
                    v_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(
                        newval - b_val[mb], -ppo_cfg.clip_coef, ppo_cfg.clip_coef
                    )
                    v_clipped = (v_clipped - b_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - ppo_cfg.ent_coef * ent_loss + ppo_cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

            if ppo_cfg.target_kl is not None and approx_kl > ppo_cfg.target_kl:
                break

        # ----------------- logging -----------------
        if update % ppo_cfg.log_every_updates == 0:
            row = {
                "update": update,
                "global_step": global_step,
                "sps": round(watch.sps(global_step)),
                "lr": optimizer.param_groups[0]["lr"],
                "ep_return_mean": float(np.mean(ep_returns)) if ep_returns else float("nan"),
                "pg_loss": float(pg_loss),
                "v_loss": float(v_loss),
                "entropy": float(ent_loss),
                "approx_kl": float(approx_kl),
                "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "policy_std": float(torch.exp(agent.actor_logstd).mean()),
            }
            (log_fn or _default_log)(row)

        if update % ppo_cfg.eval_every_updates == 0 or update == ppo_cfg.num_updates:
            ev = evaluate_policy(agent, obs_rms, env_cfg, n_episodes=5)
            (log_fn or _default_log)({"update": update, "global_step": global_step, **ev})
            if ev["eval/return_mean"] > best_eval:
                best_eval = ev["eval/return_mean"]
                save_checkpoint(ckpt_path, agent, obs_rms, env_cfg, ppo_cfg)

    envs.close()
    save_checkpoint(ckpt_path.replace(".pt", "_final.pt"), agent, obs_rms, env_cfg, ppo_cfg)
    return ckpt_path


def _default_log(row: dict) -> None:
    if "eval/return_mean" in row:
        print(
            f"  [eval] step={row['global_step']:>9,}  "
            f"return={row['eval/return_mean']:8.1f} +/- {row['eval/return_std']:5.1f}  "
            f"dist={row['eval/distance_mean']:6.2f} m  "
            f"speed={row['eval/speed_mean']:5.2f} m/s"
        )
    else:
        print(
            f"upd {row['update']:>4}  step {row['global_step']:>9,}  "
            f"{row['sps']:>5} sps  epRet {row['ep_return_mean']:8.1f}  "
            f"KL {row['approx_kl']:.3f}  std {row['policy_std']:.2f}"
        )
