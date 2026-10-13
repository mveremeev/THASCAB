"""Stage 4 null hypotheses:
B7 ensemble of solos (logit average + learned gate) per seed.
B8 monolith: depth x lr sweep on seed 0, then best config x 5 seeds."""

import json
import sys
import time

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from thascab import data as D
from thascab import models as M
from thascab import train as T

SEEDS = [0, 1, 2, 3, 4]
DEVICE = T.DEVICE
splits = T.load_splits()
t00 = time.time()


# ---------------------------------------------------------------- B7 ensemble

def load_solos(seed, b1_variant="b1ft"):
    enc = M.ViTEncoder()
    head = M.PerceiverSoloHead()
    sd = torch.load(f"runs/{b1_variant}_s{seed}/model.pt", map_location="cpu")
    enc.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("enc.")})
    head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")})
    b2 = M.Reasoner()
    b2.load_state_dict(torch.load(f"runs/b2_s{seed}/model.pt", map_location="cpu"))
    b3 = M.Interpreter()
    b3.load_state_dict(torch.load(f"runs/b3_s{seed}/model.pt", map_location="cpu"))
    return (enc.to(DEVICE).eval(), head.to(DEVICE).eval(),
            b2.to(DEVICE).eval(), b3.to(DEVICE).eval())


@torch.no_grad()
def solo_logits(models, sp, bs=512):
    enc, head, b2, b3 = models
    outs = {"b1": [], "b2": [], "b3": []}
    for i in range(0, sp.n, bs):
        idx = torch.arange(i, min(i + bs, sp.n), device=sp.device)
        b = sp.batch(idx)
        outs["b1"].append(head(enc(b["img"]), b["query"]))
        outs["b2"].append(b2(b["objects"], b["query"])[0])
        outs["b3"].append(b3(M.build_solo_seq(b["scene_seq"], b["query"])))
    return {k: torch.cat(v) for k, v in outs.items()}


def run_ensemble(seed, b1_variant):
    models = load_solos(seed, b1_variant)
    logits = {name: solo_logits(models, sp) for name, sp in splits.items()}
    res_avg, res_gate = {}, {}
    # plain average of log-probs
    for name, lg in logits.items():
        if name == "train":
            continue
        avg = sum(F.log_softmax(v, -1) for v in lg.values())
        acc = float((avg.argmax(-1) == splits[name].answer).float().mean())
        res_avg[name] = {"overall": acc}
    # learned gate on train logits
    torch.manual_seed(seed)
    gate = M.mlp(3 * len(D.ANSWERS), 64, 3).to(DEVICE)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-3)
    Xtr = torch.cat([F.log_softmax(logits["train"][k], -1) for k in ("b1", "b2", "b3")], -1)
    ytr = splits["train"].answer
    parts = [F.log_softmax(logits["train"][k], -1) for k in ("b1", "b2", "b3")]
    for step in range(600):
        w = F.softmax(gate(Xtr), -1)
        mix = sum(w[:, i:i + 1] * parts[i] for i in range(3))
        loss = F.nll_loss(F.log_softmax(mix, -1), ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        for name, lg in logits.items():
            if name == "train":
                continue
            X = torch.cat([F.log_softmax(lg[k], -1) for k in ("b1", "b2", "b3")], -1)
            p = [F.log_softmax(lg[k], -1) for k in ("b1", "b2", "b3")]
            w = F.softmax(gate(X), -1)
            mix = sum(w[:, i:i + 1] * p[i] for i in range(3))
            # per-family
            fam = splits[name].family
            ok = (mix.argmax(-1) == splits[name].answer).float()
            r = {"overall": float(ok.mean())}
            for fi, fn in enumerate(D.FAMILIES):
                m = fam == fi
                if m.any():
                    r[fn] = float(ok[m].mean())
            res_gate[name] = r
    T.save_run(f"b7_{b1_variant}_s{seed}",
               {"name": f"b7_{b1_variant}_s{seed}", "seed": seed,
                "final": res_gate, "avg_final": res_avg})
    print(f"  b7({b1_variant}) s{seed} gate iid={res_gate['iid']['overall']:.3f} "
          f"avg iid={res_avg['iid']['overall']:.3f}", flush=True)


def done(name):
    import os
    return os.path.exists(f"runs/{name}/result.json")


for seed in SEEDS:
    print(f"[B7 ensemble, seed {seed}]", flush=True)
    if not done(f"b7_b1ft_s{seed}"):
        run_ensemble(seed, "b1ft")
    if not done(f"b7_b1frozen_s{seed}"):
        run_ensemble(seed, "b1frozen")

# ---------------------------------------------------------------- B8 monolith

print("[B8 sweep on seed 0]", flush=True)
best, best_acc = None, -1
for depth in (4, 6, 8):
    for lr in (3e-4, 6e-4):
        name = f"b8sweep_d{depth}_lr{lr}"
        if done(name):
            final = json.load(open(f"runs/{name}/result.json"))["final"]
        else:
            T.set_seed(0)
            m = M.Monolith(depth=depth)
            print(f"  depth={depth} lr={lr} params={M.count_params(m)/1e6:.2f}M",
                  flush=True)
            final = T.train_loop(m, T.loss_monolith, splits, name, epochs=12,
                                 seed=0, lr=lr, eval_every=12)
        if final["iid"]["overall"] > best_acc:
            best_acc, best = final["iid"]["overall"], (depth, lr)
print(f"  best config: depth={best[0]} lr={best[1]} (iid={best_acc:.3f})", flush=True)

for seed in SEEDS:
    if done(f"b8_s{seed}"):
        continue
    print(f"[B8 monolith d={best[0]}, seed {seed}]", flush=True)
    T.set_seed(seed)
    T.train_loop(M.Monolith(depth=best[0]), T.loss_monolith, splits,
                 f"b8_s{seed}", epochs=20, seed=seed, lr=best[1])

with open("runs/b8_best_config.json", "w") as f:
    json.dump({"depth": best[0], "lr": best[1]}, f)
print(f"STAGE 4 COMPLETE in {time.time()-t00:.0f}s", flush=True)
