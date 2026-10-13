"""Training + evaluation harness for THASCAB. All runs log JSON to runs/<name>/."""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import models as M

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class Split:
    """One split held as CPU tensors; batches assembled on demand."""

    def __init__(self, path, device=DEVICE):
        z = np.load(path)
        # everything lives on-device (uint8 images; total well under 1 GB)
        self.images = torch.from_numpy(z["images"]).to(device)     # [S,64,64,3] u8
        self.objects = torch.from_numpy(z["objects"].astype(np.int64)).to(device)
        self.scene_seq = torch.from_numpy(z["scene_seq"].astype(np.int64)).to(device)
        self.scene_idx = torch.from_numpy(z["scene_idx"].astype(np.int64)).to(device)
        self.query = torch.from_numpy(z["query"].astype(np.int64)).to(device)
        self.answer = torch.from_numpy(z["answer"].astype(np.int64)).to(device)
        self.family = torch.from_numpy(z["family"].astype(np.int64)).to(device)
        self.n = len(self.answer)
        self.device = device

    def batch(self, idx, need_images=True):
        idx = idx.to(self.device)
        si = self.scene_idx[idx]
        b = {
            "objects": self.objects[si],
            "scene_seq": self.scene_seq[si],
            "query": self.query[idx],
            "answer": self.answer[idx],
            "family": self.family[idx],
        }
        if need_images:
            b["img"] = self.images[si].float().div_(255).sub_(0.5)
        return b

    def iter_batches(self, bs, shuffle=True, seed=0, need_images=True):
        order = torch.randperm(self.n, generator=torch.Generator().manual_seed(seed)) \
            if shuffle else torch.arange(self.n)
        for i in range(0, self.n, bs):
            yield self.batch(order[i:i + bs], need_images=need_images)


def load_splits(root="data/full", names=("train", "iid", "comp_attr", "comp_tmpl", "extrap")):
    return {s: Split(os.path.join(root, f"{s}.npz")) for s in names}


def cosine_lr(opt, step, total, base, warmup=300):
    lr = base * step / warmup if step < warmup else \
        base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
    return lr


@torch.no_grad()
def evaluate(forward_fn, split, bs=512, need_images=True):
    """forward_fn(batch)->logits. Returns dict overall + per-family accuracy."""
    correct, total = 0, 0
    fam_c, fam_t = np.zeros(len(D.FAMILIES)), np.zeros(len(D.FAMILIES))
    for b in split.iter_batches(bs, shuffle=False, need_images=need_images):
        pred = forward_fn(b).argmax(-1)
        ok = (pred == b["answer"]).float()
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


def eval_all(forward_fn, splits, need_images=True):
    return {name: evaluate(forward_fn, sp, need_images=need_images)
            for name, sp in splits.items() if name != "train"}


def save_run(name, payload):
    os.makedirs(f"runs/{name}", exist_ok=True)
    with open(f"runs/{name}/result.json", "w") as f:
        json.dump(payload, f, indent=2)


def train_loop(model, forward_loss, splits, name, epochs=15, bs=256, lr=3e-4,
               wd=0.01, seed=0, need_images=True, eval_every=5,
               param_groups=None, log_hook=None):
    """Generic supervised loop. forward_loss(model, batch, train)->(loss, logits)."""
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(param_groups or model.parameters(), lr=lr, weight_decay=wd)
    steps_per = math.ceil(splits["train"].n / bs)
    total = steps_per * epochs
    step, t0, hist = 0, time.time(), []
    for ep in range(epochs):
        model.train()
        for b in splits["train"].iter_batches(bs, seed=seed * 1000 + ep,
                                              need_images=need_images):
            cosine_lr(opt, step, total, lr)
            loss, _ = forward_loss(model, b, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            model.eval()
            res = eval_all(lambda bb: forward_loss(model, bb, False)[1],
                           splits, need_images=need_images)
            hist.append({"epoch": ep + 1, **{k: v["overall"] for k, v in res.items()}})
            print(f"  ep{ep+1:3d} " + " ".join(f"{k}={v['overall']:.3f}" for k, v in res.items())
                  + f" [{time.time()-t0:.0f}s]", flush=True)
            if log_hook:
                log_hook(ep + 1, model, res)
    model.eval()
    final = eval_all(lambda bb: forward_loss(model, bb, False)[1],
                     splits, need_images=need_images)
    save_run(name, {"name": name, "seed": seed, "epochs": epochs, "lr": lr,
                    "params": M.count_params(model), "history": hist,
                    "final": final, "train_time_s": time.time() - t0})
    torch.save(model.state_dict(), f"runs/{name}/model.pt")
    return final


# ---------------------------------------------------------------- solo losses

def loss_interpreter(model, b, train, scene_dropout=0.15):
    drop = None
    if train and scene_dropout > 0:
        drop = torch.rand(b["query"].shape[0], device=DEVICE) < scene_dropout
    seq = M.build_solo_seq(b["scene_seq"], b["query"], drop)
    logits = model(seq)
    return F.cross_entropy(logits, b["answer"]), logits


def loss_reasoner(model, b, train, attr_dropout=0.15, query_dropout=0.15):
    ma = dq = None
    if train:
        B = b["query"].shape[0]
        ma = torch.rand(B, device=DEVICE) < attr_dropout
        dq = torch.rand(B, device=DEVICE) < query_dropout
    logits, _, _ = model(b["objects"], b["query"], mask_attrs=ma, drop_query=dq)
    return F.cross_entropy(logits, b["answer"]), logits


def loss_monolith(model, b, train):
    logits = model(b["img"], b["scene_seq"], b["query"])
    return F.cross_entropy(logits, b["answer"]), logits


# ---------------------------------------------------------------- JEPA pretrain

def jepa_pretrain(splits, name, epochs=20, bs=256, lr=1e-3, seed=0):
    set_seed(seed)
    model = M.JEPA().to(DEVICE)
    opt = torch.optim.AdamW(model.encoder.parameters(), lr=lr, weight_decay=0.05)
    # unique scenes only (images are per-scene)
    imgs = splits["train"].images
    objs_np = splits["train"].objects.cpu().numpy()
    n = len(imgs)
    steps_per = math.ceil(n / bs)
    total = steps_per * epochs
    rng = np.random.default_rng(seed)
    opt2 = torch.optim.AdamW([p for nm, p in model.named_parameters()
                              if nm.startswith("pred")], lr=lr, weight_decay=0.05)
    step, hist, t0 = 0, [], time.time()
    for ep in range(epochs):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 77 + ep))
        model.train()
        for i in range(0, n, bs):
            sel = order[i:i + bs]
            img = imgs[sel].to(DEVICE).float().div_(255).sub_(0.5)
            vis, msk = M.block_masks_biased(objs_np[sel.numpy()], rng)
            vis_t = torch.tensor(vis, device=DEVICE)
            msk_t = torch.tensor(msk, device=DEVICE)
            cosine_lr(opt, step, total, lr)
            cosine_lr(opt2, step, total, lr)
            loss, _ = model(img, vis_t, msk_t)
            opt.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            opt2.step()
            # EMA momentum 0.996 -> 1.0 cosine
            m = 1.0 - (1.0 - 0.996) * 0.5 * (1 + math.cos(math.pi * step / total))
            model.ema_update(m)
            step += 1
        # collapse monitoring: per-dim std + effective rank on a probe batch
        model.eval()
        with torch.no_grad():
            img = imgs[:512].to(DEVICE).float().div_(255).sub_(0.5)
            z = model.encoder(img).reshape(-1, 128)
            std = z.std(0).mean().item()
            zc = (z - z.mean(0)).cpu()
            s = torch.linalg.svdvals(zc[:4096])
            p = (s / s.sum()).clamp(min=1e-9)
            erank = float(torch.exp(-(p * p.log()).sum()))
        hist.append({"epoch": ep + 1, "loss": float(loss.item()),
                     "latent_std": std, "eff_rank": erank})
        print(f"  jepa ep{ep+1:3d} loss={loss.item():.4f} std={std:.3f} "
              f"erank={erank:.1f} [{time.time()-t0:.0f}s]", flush=True)
    os.makedirs(f"runs/{name}", exist_ok=True)
    torch.save(model.encoder.state_dict(), f"runs/{name}/encoder.pt")
    save_run(name, {"name": name, "seed": seed, "history": hist,
                    "params": M.count_params(model)})
    return model.encoder


def train_perceiver_solo(splits, encoder, name, finetune=False, epochs=15,
                         seed=0, lr=3e-4):
    head = M.PerceiverSoloHead().to(DEVICE)
    encoder = encoder.to(DEVICE)
    for p in encoder.parameters():
        p.requires_grad_(finetune)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.head = encoder, head

        def forward(self, img, query):
            tok = self.enc(img) if finetune else self.enc(img).detach()
            return self.head(tok, query)

    w = Wrap()
    groups = [{"params": head.parameters()}]
    if finetune:
        groups.append({"params": encoder.parameters(), "lr_scale": 0.1})

    def fl(model, b, train):
        if not finetune:
            with torch.no_grad():
                tok = model.enc(b["img"])
            logits = model.head(tok, b["query"])
        else:
            logits = model(b["img"], b["query"])
        return F.cross_entropy(logits, b["answer"]), logits

    return train_loop(w, fl, splits, name, epochs=epochs, seed=seed, lr=lr,
                      param_groups=groups)
