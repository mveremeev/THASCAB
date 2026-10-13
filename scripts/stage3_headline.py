"""Stage 3 headline: B4 (frozen cores, cached), B4.5 (finetune from B4), B5 (scratch).
5 seeds each. B4 seed 0 also saves bridge checkpoints for observer-over-time."""

import os
import sys
import time

sys.path.insert(0, ".")

import torch

from thascab import train as T
from thascab import union_train as UT

SEEDS = [0, 1, 2, 3, 4]
splits = T.load_splits()
t00 = time.time()


def done(name):
    if os.path.exists(f"runs/{name}/result.json"):
        print(f"[skip {name}: already done]", flush=True)
        return True
    return False


for seed in SEEDS:
    if done(f"b4_s{seed}"):
        continue
    print(f"[B4 frozen union, seed {seed}]", flush=True)
    hook = None
    if seed == 0:
        os.makedirs("runs/b4_s0_ckpts", exist_ok=True)

        def hook(ep, u):
            torch.save(u.state_dict(), f"runs/b4_s0_ckpts/ep{ep}.pt")

    UT.train_union(splits, f"b4_s{seed}", seed=seed, regime="frozen",
                   epochs=25, lr=1e-3, use_cache=True, ckpt_hook=hook)

for seed in SEEDS:
    if done(f"b45_s{seed}"):
        continue
    print(f"[B4.5 finetune union, seed {seed}]", flush=True)
    UT.train_union(splits, f"b45_s{seed}", seed=seed, regime="finetune",
                   epochs=12, lr=3e-4, core_lr_scale=0.1,
                   init_bridge_from=f"runs/b4_s{seed}/model.pt")

for seed in SEEDS:
    if done(f"b5_s{seed}"):
        continue
    print(f"[B5 scratch union, seed {seed}]", flush=True)
    UT.train_union(splits, f"b5_s{seed}", seed=seed, regime="scratch",
                   epochs=20, lr=1e-3)

print(f"STAGE 3 COMPLETE in {time.time()-t00:.0f}s", flush=True)
