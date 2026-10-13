"""THASCAB union: three modules communicating through a bandwidth-limited
shared latent workspace. No router; all modules write and read every round.

Also contains the text-bridge variant (B6): slots are quantized to discrete
tokens (Gumbel-softmax, annealed) before every read — same write machinery,
so the comparison isolates latent-vs-symbolic at matched bandwidth
(K tokens/round x R rounds, 128-word message vocab)."""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D
from . import models as M

DIM = M.DIM


class Union(nn.Module):
    N_MODULES = 3  # 0=A perceiver, 1=B reasoner, 2=C interpreter

    def __init__(self, encA, gnnB, trC, K=8, R=3, freeze_cores=True,
                 text_bridge=False, msg_vocab=128, r_max=6):
        super().__init__()
        self.encA, self.gnnB, self.trC = encA, gnnB, trC
        self.K, self.R = K, R
        self.freeze_cores = freeze_cores
        self.text_bridge = text_bridge
        if freeze_cores:
            for m in (encA, gnnB, trC):
                for p in m.parameters():
                    p.requires_grad_(False)

        self.adapt = nn.ModuleList([nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM))
                                    for _ in range(3)])
        self.mod_id = nn.Parameter(torch.randn(3, DIM) * 0.02)
        self.slots0 = nn.Parameter(torch.randn(1, K, DIM) * 0.02)
        self.round_emb = nn.Parameter(torch.randn(r_max, 1, 1, DIM) * 0.02)
        self.write = nn.MultiheadAttention(DIM, 4, batch_first=True)
        self.write_ln_q = nn.LayerNorm(DIM)
        self.write_ln_k = nn.LayerNorm(DIM)
        self.gru = nn.GRUCell(DIM, DIM)
        self.read = nn.ModuleList([nn.MultiheadAttention(DIM, 4, batch_first=True)
                                   for _ in range(3)])
        self.read_ln = nn.ModuleList([nn.LayerNorm(DIM) for _ in range(3)])
        # local processing after each read (Goyal et al.: broadcast, then local
        # computation inside each module). Shared across rounds: depth grows with R.
        self.local = nn.ModuleList([M.Block(DIM, 4) for _ in range(3)])
        # answer head: learned query attends over C's tokens + final slots
        self.head_q = nn.Parameter(torch.randn(1, 1, DIM) * 0.02)
        self.head_attn = nn.MultiheadAttention(DIM, 4, batch_first=True)
        self.head_ln = nn.LayerNorm(DIM)
        self.head = nn.Sequential(nn.LayerNorm(DIM), M.mlp(DIM, DIM * 2, len(D.ANSWERS)))
        if text_bridge:
            self.q_proj = nn.Linear(DIM, msg_vocab)
            self.q_emb = nn.Embedding(msg_vocab, DIM)

    def raw_core_tokens(self, img, objects, query):
        """Partitioned union inputs: A image-only, B geometry+size only, C query-only."""
        ctx = torch.no_grad() if self.freeze_cores else nullcontext()
        with ctx:
            tokA = self.encA(img)                                     # B,64,d
            ma = torch.ones(objects.shape[0], dtype=torch.bool, device=objects.device)
            hB, padB = self.gnnB.embed_nodes(objects, mask_attrs=ma)
            eB = self.gnnB.edge_feats(objects)
            tokB = self.gnnB.message_pass(hB, eB, padB)               # B,6,d
            seqC = M.build_query_seq(query)
            tokC = self.trC.trunk(seqC)                               # B,L,d
        return tokA, tokB, tokC

    def core_tokens(self, img, objects, query, cached=None):
        """cached: optional (tokA, tokB, tokC) from precompute (frozen cores only)."""
        if cached is None:
            tokA, tokB, tokC = self.raw_core_tokens(img, objects, query)
        else:
            tokA, tokB, tokC = cached
        padA = torch.zeros(tokA.shape[:2], dtype=torch.bool, device=tokA.device)
        padB = objects[:, :, 0] < 0
        padC = M.build_query_seq(query) == D.PAD
        toks = [self.adapt[i](t) + self.mod_id[i]
                for i, t in enumerate((tokA, tokB, tokC))]
        return toks, [padA, padB, padC]

    def quantize(self, slots, tau, hard):
        logits = self.q_proj(slots)
        if hard:
            idx = logits.argmax(-1)
            return self.q_emb(idx), idx
        y = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        return y @ self.q_emb.weight, logits.argmax(-1)

    def forward(self, img, objects, query, write_mask=None, tau=1.0,
                collect=False, cached=None):
        """write_mask: [B,3] bool, True = silence that module's writes (lesion).
        collect: also return workspace internals for probing.
        cached: precomputed raw core tokens (frozen-core fast path)."""
        toks, pads = self.core_tokens(img, objects, query, cached=cached)
        B = query.shape[0]
        slots = self.slots0.expand(B, -1, -1).reshape(B * self.K, DIM)
        sizes = [t.shape[1] for t in toks]
        internals = {"slots": [], "msg_tokens": [], "write_mass": []} if collect else None
        for r in range(self.R):
            keys = torch.cat(toks, 1)
            kpm = torch.cat(pads, 1)
            if write_mask is not None:
                parts = []
                for m in range(3):
                    parts.append(pads[m] | write_mask[:, m:m + 1])
                kpm = torch.cat(parts, 1)
            q = slots.reshape(B, self.K, DIM) + self.round_emb[r]
            a, w = self.write(self.write_ln_q(q), self.write_ln_k(keys), keys,
                              key_padding_mask=kpm, need_weights=collect,
                              average_attn_weights=True)
            if collect:  # per-slot attention mass on each module's key block
                o = 0
                mass = []
                for m in range(3):
                    mass.append(w[:, :, o:o + sizes[m]].sum(-1))
                    o += sizes[m]
                internals["write_mass"].append(torch.stack(mass, -1).detach())
            slots = self.gru(a.reshape(B * self.K, DIM), slots)
            s = slots.reshape(B, self.K, DIM)
            if self.text_bridge:
                s_eff, idx = self.quantize(s, tau, hard=not self.training)
            else:
                s_eff, idx = s, None
            if collect:
                internals["slots"].append(s.detach())
                if idx is not None:
                    internals["msg_tokens"].append(idx.detach())
            new_toks = []
            for m in range(3):
                t = toks[m]
                o, _ = self.read[m](self.read_ln[m](t), s_eff, s_eff,
                                    need_weights=False)
                t = t + o
                t = self.local[m](t, key_padding_mask=pads[m])
                new_toks.append(t)
            toks = new_toks
        # only C speaks: pooled readout over C's workspace-updated tokens only
        # (slot content reaches C via reads; raw slots hold A/B writes and would
        # let them bypass the Interpreter)
        hq = self.head_q.expand(B, -1, -1)
        pooled, _ = self.head_attn(hq, self.head_ln(toks[2]), toks[2],
                                   key_padding_mask=pads[2], need_weights=False)
        logits = self.head(pooled[:, 0] + toks[2][:, -1])
        if collect:
            internals["tokens"] = [t.detach() for t in toks]
            return logits, internals
        return logits

    def bridge_parameters(self):
        core_ids = {id(p) for m in (self.encA, self.gnnB, self.trC)
                    for p in m.parameters()}
        return [p for p in self.parameters() if id(p) not in core_ids]


def load_cores(seed, device):
    """Load solo-pretrained cores for a given seed from runs/."""
    encA = M.ViTEncoder()
    encA.load_state_dict(torch.load(f"runs/jepa_s{seed}/encoder.pt",
                                    map_location="cpu"))
    gnnB = M.Reasoner()
    gnnB.load_state_dict(torch.load(f"runs/b2_s{seed}/model.pt", map_location="cpu"))
    trC = M.Interpreter()
    trC.load_state_dict(torch.load(f"runs/b3_s{seed}/model.pt", map_location="cpu"))
    return encA.to(device), gnnB.to(device), trC.to(device)
