"""Run one union config in an isolated process (MPS leak hygiene) + damage matrix."""

import argparse
import json
import sys

sys.path.insert(0, ".")

import torch

from thascab import train as T
from thascab import union_train as UT

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--regime", default="frozen")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--R", type=int, default=3)
ap.add_argument("--module_dropout", type=float, default=0.0)
ap.add_argument("--text_bridge", action="store_true")
ap.add_argument("--init_from", default=None)
ap.add_argument("--core_lr_scale", type=float, default=0.1)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--damage", action="store_true")
ap.add_argument("--ckpts", action="store_true")
args = ap.parse_args()

splits = T.load_splits()
hook = None
if args.ckpts:
    import os
    os.makedirs(f"runs/{args.name}_ckpts", exist_ok=True)

    def hook(ep, u):
        torch.save(u.state_dict(), f"runs/{args.name}_ckpts/ep{ep}.pt")

use_cache = args.regime == "frozen"
u, final, cache = UT.train_union(
    splits, args.name, seed=args.seed, regime=args.regime, K=args.K, R=args.R,
    text_bridge=args.text_bridge, epochs=args.epochs, lr=args.lr, bs=args.bs,
    core_lr_scale=args.core_lr_scale, module_dropout=args.module_dropout,
    init_bridge_from=args.init_from, eval_every=args.eval_every,
    use_cache=use_cache, ckpt_hook=hook)

if args.damage:
    u.freeze_cores = True  # eval-only guard
    dm = UT.damage_matrix(u, splits)
    json.dump(dm, open(f"runs/{args.name}/damage.json", "w"), indent=2)
    print("  damage iid:", {m: round(dm[m]["iid"]["overall"], 3) for m in dm},
          flush=True)
