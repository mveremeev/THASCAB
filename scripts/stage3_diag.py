"""Seed-0 diagnostic ladder for the union: B4 v2 -> B4.5 -> capacity variants.
Decides the config for the full Stage 3 sweep."""

import json
import sys

sys.path.insert(0, ".")

import torch

from thascab import train as T
from thascab import union_train as UT

splits = T.load_splits()

print("[1/5 b4 v2 seed0, 25ep]", flush=True)
u, f, _ = UT.train_union(splits, "b4_s0", seed=0, regime="frozen",
                         epochs=25, lr=1e-3, use_cache=True)
dm = UT.damage_matrix(u, splits)
json.dump(dm, open("runs/b4_s0/damage.json", "w"), indent=2)
print("  damage iid:", {m: round(dm[m]["iid"]["overall"], 3) for m in dm}, flush=True)
del u
torch.mps.empty_cache()

print("[2/5 b4.5 v2 seed0, 12ep]", flush=True)
u, f, _ = UT.train_union(splits, "b45_s0", seed=0, regime="finetune",
                         epochs=12, lr=3e-4, core_lr_scale=0.1,
                         init_bridge_from="runs/b4_s0/model.pt")
dm = UT.damage_matrix(u, splits)
json.dump(dm, open("runs/b45_s0/damage.json", "w"), indent=2)
print("  damage iid:", {m: round(dm[m]["iid"]["overall"], 3) for m in dm}, flush=True)
del u
torch.mps.empty_cache()

print("[3/5 b4 R=5 seed0]", flush=True)
UT.train_union(splits, "diag_r5_s0", seed=0, regime="frozen", epochs=25,
               lr=1e-3, R=5, use_cache=True, eval_every=25)
torch.mps.empty_cache()

print("[4/5 b4 K=16 seed0]", flush=True)
UT.train_union(splits, "diag_k16_s0", seed=0, regime="frozen", epochs=25,
               lr=1e-3, K=16, use_cache=True, eval_every=25)
torch.mps.empty_cache()

print("[5/5 b4 lr=2e-3 40ep seed0]", flush=True)
UT.train_union(splits, "diag_lr2e3_s0", seed=0, regime="frozen", epochs=40,
               lr=2e-3, use_cache=True, eval_every=10)
print("DIAG COMPLETE", flush=True)
