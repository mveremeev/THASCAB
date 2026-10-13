"""Stage 6: K-sweep, R-sweep (frozen+cached, 3 seeds), damage matrices,
compensation test (module-dropout training), redundancy + observer probes,
observer-over-time on B4 seed-0 checkpoints."""

import json
import sys
import time

sys.path.insert(0, ".")

import torch

from thascab import probes as P
from thascab import train as T
from thascab import union_train as UT

SWEEP_SEEDS = [0, 1, 2]
ALL_SEEDS = [0, 1, 2, 3, 4]
splits = T.load_splits()
t00 = time.time()

# ---- K-sweep and R-sweep (K=8,R=3 baseline = b4 runs) ----
for K in (1, 2, 4, 16):
    for seed in SWEEP_SEEDS:
        print(f"[K-sweep K={K}, seed {seed}]", flush=True)
        UT.train_union(splits, f"ksweep_k{K}_s{seed}", seed=seed, regime="frozen",
                       epochs=25, lr=1e-3, K=K, use_cache=True, eval_every=25)

for R in (1, 2, 5):
    for seed in SWEEP_SEEDS:
        print(f"[R-sweep R={R}, seed {seed}]", flush=True)
        UT.train_union(splits, f"rsweep_r{R}_s{seed}", seed=seed, regime="frozen",
                       epochs=25, lr=1e-3, R=R, use_cache=True, eval_every=25)

# ---- compensation variant: frozen + module dropout on writes ----
for seed in SWEEP_SEEDS:
    print(f"[B4+dropout, seed {seed}]", flush=True)
    UT.train_union(splits, f"b4drop_s{seed}", seed=seed, regime="frozen",
                   epochs=25, lr=1e-3, use_cache=True, module_dropout=0.3,
                   eval_every=25)

# ---- damage matrices: B4, B4.5, B4drop, B5 ----
def dm_for(name, seed, regime):
    u = UT.build_union(seed, "frozen" if regime != "scratch" else "scratch")
    sd = torch.load(f"runs/{name}/model.pt", map_location="cpu")
    u.load_state_dict(sd)
    u.to(UT.DEVICE).eval()
    u.freeze_cores = True  # eval only
    dm = UT.damage_matrix(u, splits)
    with open(f"runs/{name}/damage.json", "w") as f:
        json.dump(dm, f, indent=2)
    drop = {m: round(dm["none"]["iid"]["overall"] - dm[m]["iid"]["overall"], 3)
            for m in ("A_perceiver", "B_reasoner", "C_interpreter")}
    print(f"  {name}: iid drops {drop}", flush=True)
    return u

for seed in ALL_SEEDS:
    dm_for(f"b4_s{seed}", seed, "frozen")
    dm_for(f"b45_s{seed}", seed, "finetune")
for seed in SWEEP_SEEDS:
    dm_for(f"b4drop_s{seed}", seed, "frozen")
    dm_for(f"b5_s{seed}", seed, "scratch")

# ---- probes: observer + redundancy on B4 & B4.5 seed 0 ----
def probe_run(name, seed):
    u = UT.build_union(seed, "frozen")
    u.load_state_dict(torch.load(f"runs/{name}/model.pt", map_location="cpu"))
    u.to(UT.DEVICE).eval()
    tr = P.collect_states(u, splits["train"], "train", n_max=8000)
    te = P.collect_states(u, splits["iid"], "iid", n_max=6000)
    rep = {"observer": P.observer_report(tr, te, u.K),
           "redundancy": P.redundancy_report(tr, te)}
    with open(f"runs/{name}/probes.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"  {name} observer: " + " ".join(
        f"{k}={v:.3f}" for k, v in rep["observer"].items()
        if isinstance(v, float)), flush=True)

probe_run("b4_s0", 0)
probe_run("b45_s0", 0)

# ---- observer decodability over training time (B4 seed-0 checkpoints) ----
curve = []
for ep in (5, 10, 15, 20, 25):
    u = UT.build_union(0, "frozen")
    u.load_state_dict(torch.load(f"runs/b4_s0_ckpts/ep{ep}.pt", map_location="cpu"))
    u.to(UT.DEVICE).eval()
    tr = P.collect_states(u, splits["train"], "train", n_max=6000)
    te = P.collect_states(u, splits["iid"], "iid", n_max=4000)
    rep = P.observer_report(tr, te, u.K)
    curve.append({"epoch": ep, **{k: v for k, v in rep.items()
                                  if isinstance(v, float)}})
    print(f"  observer@ep{ep}: {curve[-1]}", flush=True)
with open("runs/observer_curve.json", "w") as f:
    json.dump(curve, f, indent=2)

print(f"STAGE 6 COMPLETE in {time.time()-t00:.0f}s", flush=True)
