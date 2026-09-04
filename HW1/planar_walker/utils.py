"""Small helpers: seeding, running mean/std normalizers, logging."""
from __future__ import annotations

import random
import time
from typing import Dict

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class RunningMeanStd:
    """Welford-style running estimate of mean/variance (per element).

    Used to normalize observations (and, indirectly, returns) during training.
    Parallel update rule from
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float | None = None) -> np.ndarray:
        out = (x - self.mean) / np.sqrt(self.var + 1e-8)
        if clip is not None:
            out = np.clip(out, -clip, clip)
        return out.astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean.copy(), "var": self.var.copy(),
                "count": np.array(self.count)}

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(sd["mean"], dtype=np.float64)
        self.var = np.asarray(sd["var"], dtype=np.float64)
        self.count = float(sd["count"])


class ReturnNormalizer:
    """Normalizes rewards by a running estimate of the std of the discounted return.

    This is the standard trick used by Stable-Baselines3 / CleanRL: it keeps the
    value-function targets at a sane scale without changing the sign or the
    relative shape of the reward.
    """

    def __init__(self, num_envs: int, gamma: float):
        self.rms = RunningMeanStd(shape=())
        self.ret = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + reward
        self.rms.update(self.ret)
        self.ret[done.astype(bool)] = 0.0
        return (reward / np.sqrt(self.rms.var + 1e-8)).astype(np.float32)

    def state_dict(self):
        return self.rms.state_dict()

    def load_state_dict(self, sd):
        self.rms.load_state_dict(sd)


class Stopwatch:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def sps(self, steps: int) -> float:
        return steps / max(self.elapsed(), 1e-9)
