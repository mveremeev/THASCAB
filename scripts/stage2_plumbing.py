"""Stage 2: plumbing check on the single-hop curriculum task.
1) gradient-flow assertions through every bridge component
2) from-scratch union must at least match B3 solo on the easy task."""

import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from thascab import models as M
from thascab import train as T
from thascab import union_train as UT

splits = T.load_splits("data/curriculum", names=("train", "iid"))
print("curriculum:", splits["train"].n, "train pairs")

# ---- 1) gradient flow ----
T.set_seed(0)
u = UT.build_union(0, "scratch", K=8, R=3)
b = splits["train"].batch(torch.arange(64))
logits = u(b["img"], b["objects"], b["query"])
loss = F.cross_entropy(logits, b["answer"])
loss.backward()

checks = {
    "slots0": u.slots0, "write.in_proj": u.write.in_proj_weight,
    "gru.weight_ih": u.gru.weight_ih, "mod_id": u.mod_id,
    "adapt_A": u.adapt[0][1].weight, "adapt_B": u.adapt[1][1].weight,
    "adapt_C": u.adapt[2][1].weight,
    "read_A": u.read[0].in_proj_weight, "read_B": u.read[1].in_proj_weight,
    "read_C": u.read[2].in_proj_weight,
    "coreA(first ViT block)": u.encA.blocks[0].attn.in_proj_weight,
    "coreB(msg mlp)": u.gnnB.msg[0][0].weight,
    "coreC(first block)": u.trC.blocks[0].attn.in_proj_weight,
}
ok = True
for n, p in checks.items():
    g = 0.0 if p.grad is None else float(p.grad.norm())
    status = "OK" if g > 1e-9 else "!! ZERO GRAD"
    if g <= 1e-9:
        ok = False
    print(f"  grad {n:24s} {g:.2e} {status}")
assert ok, "gradient flow broken"
print("gradient flow: ALL OK\n")

# ---- 2) easy-task accuracy: scratch union vs B3 solo ----
print("[B3 solo on curriculum]")
T.set_seed(0)
b3 = T.train_loop(M.Interpreter(), T.loss_interpreter, splits, "cur_b3_s0",
                  epochs=5, seed=0, need_images=False, eval_every=5)
print("[scratch union on curriculum]")
_, uf, _ = UT.train_union(splits, "cur_union_s0", seed=0, regime="scratch",
                          epochs=10, eval_every=5)
print(f"\nplumbing verdict: B3 iid={b3['iid']['overall']:.3f}  "
      f"union iid={uf['iid']['overall']:.3f}")
