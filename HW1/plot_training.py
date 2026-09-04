#!/usr/bin/env python3
"""Parse a training log (stdout of train.py) into a learning-curve figure.

    python train.py ... | tee checkpoints/train_log.txt
    python plot_training.py checkpoints/train_log.txt --out curve.png
"""
from __future__ import annotations

import argparse
import re

import matplotlib.pyplot as plt

UPD = re.compile(r"step\s+([\d,]+).*?epRet\s+([-\d.]+).*?KL\s+([\d.]+).*?std\s+([\d.]+)")
EVAL = re.compile(r"\[eval\]\s+step=\s*([\d,]+)\s+return=\s*([-\d.]+).*?dist=\s*([-\d.]+)\s*m\s+speed=\s*([-\d.]+)")


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--out", default="training_curve.png")
    a = p.parse_args()

    tr, ev = [], []
    for line in open(a.log):
        if (m := EVAL.search(line)):
            ev.append((_int(m[1]), float(m[2]), float(m[3]), float(m[4])))
        elif (m := UPD.search(line)):
            tr.append((_int(m[1]), float(m[2]), float(m[3]), float(m[4])))

    if not tr:
        raise SystemExit("no training rows parsed -- is this a train.py log?")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    xs = [r[0] for r in tr]
    ax[0].plot(xs, [r[1] for r in tr], lw=1, label="train epRet")
    if ev:
        ax[0].plot([e[0] for e in ev], [e[1] for e in ev], "o-", label="eval return")
    ax[0].set_title("return"); ax[0].set_xlabel("env steps"); ax[0].legend()

    if ev:
        ax[1].plot([e[0] for e in ev], [e[2] for e in ev], "s-", color="tab:green")
        ax[1].set_title("eval distance [m] per episode"); ax[1].set_xlabel("env steps")

    ax[2].plot(xs, [r[2] for r in tr], label="approx KL")
    ax[2].plot(xs, [r[3] for r in tr], label="policy std")
    ax[2].set_title("optimization health"); ax[2].set_xlabel("env steps"); ax[2].legend()

    plt.tight_layout()
    plt.savefig(a.out, dpi=120)
    print(f"wrote {a.out}  ({len(tr)} updates, {len(ev)} evals)")


if __name__ == "__main__":
    main()
