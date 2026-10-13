"""Interpretability: passive observer probes on the workspace (stop-gradient by
construction: probes train post-hoc on detached activations) and redundancy
probes on module tokens after communication."""

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import models as M
from . import train as T

DEVICE = T.DEVICE
RED = D.COLORS.index("red")


@torch.no_grad()
def collect_states(u, sp, name, cache=None, n_max=6000, bs=512):
    """Run union with collect=True; gather workspace + targets (all detached)."""
    u.eval()
    n = min(sp.n, n_max)
    out = {k: [] for k in ["slots", "write_mass", "tokA", "tokB", "tokC",
                           "answer", "family", "n_obj", "count_red"]}
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=sp.device)
        b = sp.batch(idx, need_images=cache is None)
        cached = cache.fetch(name, sp.scene_idx[idx], idx) if cache else None
        _, ints = u(b.get("img"), b["objects"], b["query"], collect=True,
                    cached=cached, tau=0.5)
        out["slots"].append(ints["slots"][-1].cpu())          # final round
        out["write_mass"].append(ints["write_mass"][-1].cpu())
        valid = (b["objects"][:, :, 0] >= 0).float()[..., None]
        out["tokA"].append(ints["tokens"][0].mean(1).cpu())
        out["tokB"].append(((ints["tokens"][1] * valid).sum(1)
                            / valid.sum(1)).cpu())
        out["tokC"].append(ints["tokens"][2][:, -1].cpu())    # <q> position
        out["answer"].append(b["answer"].cpu())
        out["family"].append(b["family"].cpu())
        out["n_obj"].append((b["objects"][:, :, 0] >= 0).sum(1).cpu())
        out["count_red"].append((b["objects"][:, :, 3] == RED).sum(1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def probe_acc(Xtr, ytr, Xte, yte, n_cls, steps=400, lr=1e-2, hidden=0):
    """Linear (or 1-hidden) probe; returns test accuracy."""
    torch.manual_seed(0)
    Xtr, ytr = Xtr.float().to(DEVICE), ytr.long().to(DEVICE)
    Xte, yte = Xte.float().to(DEVICE), yte.long().to(DEVICE)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    net = (M.mlp(Xtr.shape[1], hidden, n_cls) if hidden
           else torch.nn.Linear(Xtr.shape[1], n_cls)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    for s in range(steps):
        loss = F.cross_entropy(net(Xtr), ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((net(Xte).argmax(-1) == yte).float().mean())


def observer_report(states_tr, states_te, K):
    """Decode task factors from workspace slots; writer identity per slot."""
    flat_tr = states_tr["slots"].reshape(len(states_tr["slots"]), -1)
    flat_te = states_te["slots"].reshape(len(states_te["slots"]), -1)
    rep = {}
    rep["answer"] = probe_acc(flat_tr, states_tr["answer"], flat_te,
                              states_te["answer"], len(D.ANSWERS))
    rep["family"] = probe_acc(flat_tr, states_tr["family"], flat_te,
                              states_te["family"], len(D.FAMILIES))
    rep["n_obj"] = probe_acc(flat_tr, states_tr["n_obj"] - 3, flat_te,
                             states_te["n_obj"] - 3, 4)
    rep["count_red"] = probe_acc(flat_tr, states_tr["count_red"], flat_te,
                                 states_te["count_red"], 7)
    # writer identity: per-slot, which module got max write attention (final round)
    wtr = states_tr["write_mass"].argmax(-1).reshape(-1)
    wte = states_te["write_mass"].argmax(-1).reshape(-1)
    str_ = states_tr["slots"].reshape(-1, M.DIM)
    ste = states_te["slots"].reshape(-1, M.DIM)
    rep["writer_id"] = probe_acc(str_, wtr, ste, wte, 3)
    rep["writer_share"] = [float((wte == m).float().mean()) for m in range(3)]
    return rep


def redundancy_report(states_tr, states_te):
    """Can info that entered via another module be decoded from a module's own
    post-communication tokens? (B never saw color; A never saw the query; C
    never saw the scene.)"""
    rep = {}
    for mod, tok in [("A", "tokA"), ("B", "tokB"), ("C", "tokC")]:
        rep[f"{mod}_answer"] = probe_acc(states_tr[tok], states_tr["answer"],
                                         states_te[tok], states_te["answer"],
                                         len(D.ANSWERS))
        rep[f"{mod}_count_red"] = probe_acc(states_tr[tok], states_tr["count_red"],
                                            states_te[tok], states_te["count_red"], 7)
        rep[f"{mod}_family"] = probe_acc(states_tr[tok], states_tr["family"],
                                         states_te[tok], states_te["family"],
                                         len(D.FAMILIES))
    return rep
