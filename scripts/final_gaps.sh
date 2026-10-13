#!/bin/zsh
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
echo "[gap 1: b8 seeds 3,4]"
python3 - <<'PY' || exit 1
import sys, json
sys.path.insert(0, ".")
from thascab import train as T, models as M
cfg = json.load(open("runs/b8_best_config.json")) if __import__('os').path.exists("runs/b8_best_config.json") else {"depth": 6, "lr": 6e-4}
splits = T.load_splits()
for seed in (3, 4):
    import os
    if os.path.exists(f"runs/b8_s{seed}/result.json"): continue
    print(f"[B8 seed {seed}]", flush=True)
    T.set_seed(seed)
    T.train_loop(M.Monolith(depth=cfg["depth"]), T.loss_monolith, splits,
                 f"b8_s{seed}", epochs=20, seed=seed, lr=cfg["lr"])
PY
echo "[gap 2: b4_s0 ckpt rerun for observer curve]"
python3 scripts/run_union_cfg.py --name b4obs_s0 --epochs 25 --ckpts || exit 1
echo "[gap 3: probes + observer curve]"
python3 - <<'PY' || exit 1
import sys, json, torch
sys.path.insert(0, ".")
from thascab import train as T, union_train as UT, probes as P
splits = T.load_splits()
def probe_run(name, regime):
    u = UT.build_union(0, "frozen")
    u.load_state_dict(torch.load(f"runs/{name}/model.pt", map_location="cpu"))
    u.to(UT.DEVICE).eval(); u.freeze_cores = True
    tr = P.collect_states(u, splits["train"], "train", n_max=8000)
    te = P.collect_states(u, splits["iid"], "iid", n_max=6000)
    rep = {"observer": P.observer_report(tr, te, u.K),
           "redundancy": P.redundancy_report(tr, te)}
    json.dump(rep, open(f"runs/{name}/probes.json", "w"), indent=2)
    print(name, {k: round(v,3) for k,v in rep["observer"].items() if isinstance(v,float)}, flush=True)
probe_run("b4_s0", "frozen")
probe_run("b45_s0", "finetune")
curve = []
for ep in (5, 10, 15, 20, 25):
    u = UT.build_union(0, "frozen")
    u.load_state_dict(torch.load(f"runs/b4obs_s0_ckpts/ep{ep}.pt", map_location="cpu"))
    u.to(UT.DEVICE).eval(); u.freeze_cores = True
    tr = P.collect_states(u, splits["train"], "train", n_max=6000)
    te = P.collect_states(u, splits["iid"], "iid", n_max=4000)
    rep = P.observer_report(tr, te, u.K)
    curve.append({"epoch": ep, **{k: v for k, v in rep.items() if isinstance(v, float)},
                  "writer_share": rep["writer_share"]})
    print("observer ep", ep, curve[-1], flush=True)
json.dump(curve, open("runs/observer_curve.json", "w"), indent=2)
PY
echo "FINAL GAPS COMPLETE"
