"""Training for the THASCAB union (B4 frozen / B4.5 finetune / B5 scratch / B6 text)."""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import models as M
from . import train as T
from .bridge import Union, load_cores

DEVICE = T.DEVICE


class CoreCache:
    """Precomputed frozen-core activations, held on CPU (fp16) to keep the MPS
    pool small; unified memory makes per-batch transfer cheap.
    tokA/tokB are per-scene, tokC per-pair. Only valid for frozen cores."""

    def __init__(self, u, splits, bs=512):
        assert u.freeze_cores
        self.store = {}
        u.eval()
        with torch.no_grad():
            for name, sp in splits.items():
                S = sp.images.shape[0]
                tokA = torch.empty(S, 64, M.DIM, dtype=torch.float16)
                tokB = torch.empty(S, 6, M.DIM, dtype=torch.float16)
                for i in range(0, S, bs):
                    img = sp.images[i:i + bs].float().div_(255).sub_(0.5)
                    obj = sp.objects[i:i + bs]
                    tokA[i:i + bs] = u.encA(img).half().cpu()
                    ma = torch.ones(obj.shape[0], dtype=torch.bool, device=DEVICE)
                    hB, padB = u.gnnB.embed_nodes(obj, mask_attrs=ma)
                    tokB[i:i + bs] = u.gnnB.message_pass(hB, u.gnnB.edge_feats(obj),
                                                         padB).half().cpu()
                N = sp.n
                L = D.SCENE_LEN + D.Q_LEN + 1
                tokC = torch.empty(N, L, M.DIM, dtype=torch.float16)
                for i in range(0, N, bs):
                    seq = M.build_query_seq(sp.query[i:i + bs])
                    tokC[i:i + bs] = u.trC.trunk(seq).half().cpu()
                self.store[name] = (tokA, tokB, tokC)
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    def fetch(self, name, scene_idx, pair_idx):
        tokA, tokB, tokC = self.store[name]
        si, pi = scene_idx.cpu(), pair_idx.cpu()
        return (tokA[si].to(DEVICE).float(), tokB[si].to(DEVICE).float(),
                tokC[pi].to(DEVICE).float())


def evaluate_union(u, sp, name, cache=None, tau=0.5, write_mask_mod=None, bs=512):
    """Per-family accuracy for one split; optional cache and lesion."""
    correct, total = 0, 0
    fam_c = np.zeros(len(D.FAMILIES))
    fam_t = np.zeros(len(D.FAMILIES))
    with torch.no_grad():
        for i in range(0, sp.n, bs):
            idx = torch.arange(i, min(i + bs, sp.n), device=sp.device)
            b = sp.batch(idx, need_images=cache is None)
            cached = cache.fetch(name, sp.scene_idx[idx], idx) if cache else None
            wm = None
            if write_mask_mod is not None:
                wm = torch.zeros(len(idx), 3, dtype=torch.bool, device=DEVICE)
                wm[:, write_mask_mod] = True
            logits = u(b.get("img"), b["objects"], b["query"], write_mask=wm,
                       tau=tau, cached=cached)
            ok = (logits.argmax(-1) == b["answer"]).float()
            correct += ok.sum().item()
            total += len(ok)
            for f in b["family"].unique():
                m = b["family"] == f
                fam_c[f] += ok[m].sum().item()
                fam_t[f] += m.sum().item()
    out = {"overall": correct / total}
    for i, f in enumerate(D.FAMILIES):
        if fam_t[i]:
            out[f] = fam_c[i] / fam_t[i]
    return out


def eval_union_all(u, splits, cache=None, tau=0.5, write_mask_mod=None):
    return {n: evaluate_union(u, sp, n, cache, tau, write_mask_mod)
            for n, sp in splits.items() if n != "train"}


def make_write_mask(B, p, device):
    """With prob p per sample, silence one uniformly chosen module's writes."""
    if p <= 0:
        return None
    drop = torch.rand(B, device=device) < p
    which = torch.randint(0, 3, (B,), device=device)
    return F.one_hot(which, 3).bool() & drop[:, None]


def build_union(seed, regime, K=8, R=3, text_bridge=False, init_bridge_from=None):
    """regime: frozen | finetune | scratch."""
    if regime == "scratch":
        cores = (M.ViTEncoder().to(DEVICE), M.Reasoner().to(DEVICE),
                 M.Interpreter().to(DEVICE))
        u = Union(*cores, K=K, R=R, freeze_cores=False, text_bridge=text_bridge)
    else:
        cores = load_cores(seed, DEVICE)
        u = Union(*cores, K=K, R=R, freeze_cores=(regime == "frozen"),
                  text_bridge=text_bridge)
    if init_bridge_from is not None:
        sd = torch.load(init_bridge_from, map_location="cpu")
        u.load_state_dict(sd, strict=False)
        if regime == "finetune":  # re-enable core grads after frozen-B4 init
            for m in (u.encA, u.gnnB, u.trC):
                for p in m.parameters():
                    p.requires_grad_(True)
            u.freeze_cores = False
    return u.to(DEVICE)


def train_union(splits, name, seed, regime, K=8, R=3, text_bridge=False,
                epochs=20, bs=256, lr=1e-3, core_lr_scale=0.1,
                module_dropout=0.0, init_bridge_from=None, eval_every=5,
                tau_hi=5.0, tau_lo=0.5, ckpt_hook=None, use_cache=False,
                cache=None):
    T.set_seed(seed)
    u = build_union(seed, regime, K, R, text_bridge, init_bridge_from)
    built_cache = False
    if use_cache:
        assert regime == "frozen"
        if cache is None:
            tc = time.time()
            cache = CoreCache(u, splits)
            built_cache = True
            print(f"  core cache built [{time.time()-tc:.0f}s]", flush=True)
    bridge_params = u.bridge_parameters()
    groups = [{"params": [p for p in bridge_params if p.requires_grad]}]
    if regime in ("finetune", "scratch"):
        core_params = [p for m in (u.encA, u.gnnB, u.trC) for p in m.parameters()]
        groups.append({"params": core_params,
                       "lr_scale": core_lr_scale if regime == "finetune" else 1.0})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.01)
    sp = splits["train"]
    steps_per = math.ceil(sp.n / bs)
    total = steps_per * epochs
    step, t0, hist = 0, time.time(), []

    for ep in range(epochs):
        u.train()
        if u.freeze_cores:
            u.encA.eval(), u.gnnB.eval(), u.trC.eval()
        order = torch.randperm(sp.n, generator=torch.Generator().manual_seed(
            seed * 1000 + ep)).to(sp.device)
        for i in range(0, sp.n, bs):
            idx = order[i:i + bs]
            if len(idx) < bs:
                continue  # uniform shapes: MPSGraph caches a kernel per shape
            b = sp.batch(idx, need_images=cache is None)
            cached = cache.fetch("train", sp.scene_idx[idx], idx) if cache else None
            T.cosine_lr(opt, step, total, lr)
            tau = tau_lo + (tau_hi - tau_lo) * 0.5 * (1 + math.cos(math.pi * step / total)) \
                if text_bridge else 1.0
            wm = make_write_mask(len(idx), module_dropout, DEVICE)
            logits = u(b.get("img"), b["objects"], b["query"], write_mask=wm,
                       tau=tau, cached=cached)
            loss = F.cross_entropy(logits, b["answer"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            # MPSGraph workspace accumulates on full-core backward; flush often
            if DEVICE.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            u.eval()
            res = eval_union_all(u, splits, cache, tau_lo)
            hist.append({"epoch": ep + 1, **{k: v["overall"] for k, v in res.items()}})
            print(f"  ep{ep+1:3d} " + " ".join(f"{k}={v['overall']:.3f}"
                  for k, v in res.items()) + f" [{time.time()-t0:.0f}s]", flush=True)
            if ckpt_hook:
                ckpt_hook(ep + 1, u)
    u.eval()
    final = eval_union_all(u, splits, cache, tau_lo)
    n_bridge = sum(p.numel() for p in u.bridge_parameters())
    T.save_run(name, {"name": name, "seed": seed, "regime": regime, "K": K, "R": R,
                      "text_bridge": text_bridge, "epochs": epochs, "lr": lr,
                      "module_dropout": module_dropout,
                      "bridge_params": n_bridge, "total_params": M.count_params(u),
                      "history": hist, "final": final,
                      "train_time_s": time.time() - t0})
    torch.save(u.state_dict(), f"runs/{name}/model.pt")
    # memory hygiene: MPS allocator hoards freed blocks; release the cache pool
    if built_cache:
        del cache
        cache = None
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return u, final, cache


@torch.no_grad()
def damage_matrix(u, splits, cache=None, tau=0.5):
    """Causal write-masking: silence each module in turn, per-family accuracy."""
    u.eval()
    out = {"none": eval_union_all(u, splits, cache, tau)}
    for mi, mname in enumerate(["A_perceiver", "B_reasoner", "C_interpreter"]):
        out[mname] = eval_union_all(u, splits, cache, tau, write_mask_mod=mi)
    return out
