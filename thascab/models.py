"""THASCAB modules: Perceiver (JEPA), Reasoner (GNN), Interpreter (AR transformer),
plus the parameter-matched Monolith baseline (B8)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D

DIM = 128


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, dim=DIM, heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = mlp(dim, dim * 4, dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


# ================================================================ A: Perceiver (JEPA)

class ViTEncoder(nn.Module):
    """64x64x3 -> 64 patch tokens (8x8 patches), 4 blocks, dim 128."""

    def __init__(self, dim=DIM, depth=4, heads=4):
        super().__init__()
        self.proj = nn.Linear(8 * 8 * 3, dim)
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)

    def patchify(self, img):  # img [B,64,64,3] float
        B = img.shape[0]
        p = img.unfold(1, 8, 8).unfold(2, 8, 8)          # B,8,8,3,8,8
        p = p.permute(0, 1, 2, 4, 5, 3).reshape(B, 64, 192)
        return p

    def forward(self, img, visible_idx=None):
        """visible_idx: [B, V] long — encode only these patches (context encoder)."""
        x = self.proj(self.patchify(img)) + self.pos
        if visible_idx is not None:
            x = torch.gather(x, 1, visible_idx[..., None].expand(-1, -1, x.shape[-1]))
        for b in self.blocks:
            x = b(x)
        return self.ln(x)


class JEPA(nn.Module):
    """I-JEPA-style: context encoder + EMA target encoder + light predictor
    (one cross-attention from masked-position queries into context + 2-layer MLP)."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.encoder = ViTEncoder(dim)
        self.target = ViTEncoder(dim)
        self.target.load_state_dict(self.encoder.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.pred_pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        self.pred_xattn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.pred_ln = nn.LayerNorm(dim)
        self.pred_mlp = mlp(dim, dim * 2, dim)

    @torch.no_grad()
    def ema_update(self, m):
        for pt, pe in zip(self.target.parameters(), self.encoder.parameters()):
            pt.mul_(m).add_(pe.detach(), alpha=1 - m)

    def forward(self, img, visible_idx, masked_idx):
        ctx = self.encoder(img, visible_idx)                       # B,V,d
        with torch.no_grad():
            tgt_all = self.target(img)                             # B,64,d
            tgt = torch.gather(tgt_all, 1,
                               masked_idx[..., None].expand(-1, -1, tgt_all.shape[-1]))
        q = self.pred_pos.expand(img.shape[0], -1, -1)
        q = torch.gather(q, 1, masked_idx[..., None].expand(-1, -1, q.shape[-1]))
        h, _ = self.pred_xattn(q, ctx, ctx, need_weights=False)
        pred = self.pred_mlp(self.pred_ln(h + q))
        return F.smooth_l1_loss(pred, tgt), ctx


def block_masks_biased(objects_np, rng, n_mask=28, n_grid=8, p_obj=0.75):
    """Per-sample block masks biased toward object locations (sparse scenes make
    uniform masks trivially predictable background). objects_np: [B,6,5] int.
    Returns (visible [B,64-n_mask], masked [B,n_mask]) index arrays."""
    import numpy as np
    B = objects_np.shape[0]
    vis = np.empty((B, n_grid * n_grid - n_mask), np.int64)
    msk = np.empty((B, n_mask), np.int64)
    for b in range(B):
        centers = []
        for o in objects_np[b]:
            if o[0] < 0:
                continue
            px, py = (o[0] + 0.5) * 64 / 6, (o[1] + 0.5) * 64 / 6
            centers.append((min(int(px // 8), n_grid - 1),
                            min(int(py // 8), n_grid - 1)))
        cells = set()
        tries = 0
        while len(cells) < n_mask and tries < 24:
            tries += 1
            if centers and rng.random() < p_obj:
                cx, cy = centers[rng.integers(len(centers))]
                w, h = int(rng.integers(2, 4)), int(rng.integers(2, 4))
                x0 = int(np.clip(cx - w // 2 + rng.integers(-1, 2), 0, n_grid - w))
                y0 = int(np.clip(cy - h // 2 + rng.integers(-1, 2), 0, n_grid - h))
            else:
                w, h = int(rng.integers(2, 5)), int(rng.integers(2, 5))
                x0 = int(rng.integers(0, n_grid - w + 1))
                y0 = int(rng.integers(0, n_grid - h + 1))
            for yy in range(y0, y0 + h):
                for xx in range(x0, x0 + w):
                    cells.add(yy * n_grid + xx)
        cells = list(cells)
        if len(cells) > n_mask:
            perm = rng.permutation(len(cells))
            cells = [cells[i] for i in perm[:n_mask]]
        elif len(cells) < n_mask:
            rest = [i for i in range(n_grid * n_grid) if i not in set(cells)]
            perm = rng.permutation(len(rest))
            cells += [rest[i] for i in perm[:n_mask - len(cells)]]
        cs = set(cells)
        msk[b] = sorted(cs)
        vis[b] = [i for i in range(n_grid * n_grid) if i not in cs]
    return vis, msk


def block_mask(rng, n_grid=8, lo=0.4, hi=0.6):
    """Sample rectangular blocks on the 8x8 patch grid covering lo..hi of patches.
    Returns (visible_idx, masked_idx) as sorted lists."""
    target = rng.uniform(lo, hi) * n_grid * n_grid
    masked = set()
    for _ in range(8):
        if len(masked) >= target:
            break
        w = rng.integers(2, 5)
        h = rng.integers(2, 5)
        x0 = rng.integers(0, n_grid - w + 1)
        y0 = rng.integers(0, n_grid - h + 1)
        for yy in range(y0, y0 + h):
            for xx in range(x0, x0 + w):
                masked.add(yy * n_grid + xx)
    masked = sorted(masked)
    if len(masked) == n_grid * n_grid:  # keep at least 8 visible
        masked = masked[:-8]
    visible = [i for i in range(n_grid * n_grid) if i not in set(masked)]
    return visible, masked


class PerceiverSoloHead(nn.Module):
    """B1: frozen (or finetuned) JEPA encoder + query cross-attention head."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.qemb = nn.Embedding(D.VOCAB_SIZE, dim)
        self.qpos = nn.Parameter(torch.randn(1, D.Q_LEN, dim) * 0.02)
        self.qenc = nn.ModuleList([Block(dim) for _ in range(2)])
        self.xattn = nn.ModuleList([nn.MultiheadAttention(dim, 4, batch_first=True)
                                    for _ in range(2)])
        self.lns = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.head = mlp(dim, dim * 2, len(D.ANSWERS))

    def forward(self, patch_tokens, query):
        pad = query == D.PAD
        q = self.qemb(query) + self.qpos
        for b in self.qenc:
            q = b(q, key_padding_mask=pad)
        s = q[:, :1]  # first token as summary query
        for ln, xa in zip(self.lns, self.xattn):
            a, _ = xa(ln(s), patch_tokens, patch_tokens, need_weights=False)
            s = s + a
        return self.head(s[:, 0])


# ================================================================ B: Reasoner (GNN)

N_X, N_SHAPE, N_COLOR, N_SIZE = D.GRID, 3, 3, 2  # + MASK slot appended


class Reasoner(nn.Module):
    """Fully-connected message passing over objects; query as global node (solo).
    Union mode: shape/color fed as MASK, no query node."""

    def __init__(self, dim=DIM, rounds=4):
        super().__init__()
        self.rounds = rounds
        self.ex = nn.Embedding(N_X + 1, dim)      # last index = MASK
        self.ey = nn.Embedding(N_X + 1, dim)
        self.es = nn.Embedding(N_SHAPE + 1, dim)
        self.ec = nn.Embedding(N_COLOR + 1, dim)
        self.ez = nn.Embedding(N_SIZE + 1, dim)
        self.qemb = nn.Embedding(D.VOCAB_SIZE, dim)
        self.qmlp = mlp(dim, dim * 2, dim)
        self.edge = mlp(3, 64, dim)
        self.msg = nn.ModuleList([mlp(dim * 3, dim * 2, dim) for _ in range(rounds)])
        self.upd = nn.ModuleList([mlp(dim * 2, dim * 2, dim) for _ in range(rounds)])
        self.pool_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = mlp(dim, dim * 2, len(D.ANSWERS))

    def embed_nodes(self, objects, mask_attrs=None):
        """objects [B,6,5] int (-1 pad). mask_attrs: [B] bool or None.
        Returns node feats [B,6,d], pad mask [B,6]."""
        pad = objects[:, :, 0] < 0
        o = objects.clamp(min=0).long()
        sh, co = o[:, :, 2], o[:, :, 3]
        if mask_attrs is not None:
            m = mask_attrs[:, None]
            sh = torch.where(m, torch.full_like(sh, N_SHAPE), sh)
            co = torch.where(m, torch.full_like(co, N_COLOR), co)
        h = (self.ex(o[:, :, 0]) + self.ey(o[:, :, 1]) + self.es(sh)
             + self.ec(co) + self.ez(o[:, :, 4]))
        return h, pad

    def edge_feats(self, objects):
        xy = objects[:, :, :2].float()            # B,6,2
        d = xy[:, :, None, :] - xy[:, None, :, :]  # B,6,6,2
        dist = d.norm(dim=-1, keepdim=True)
        return self.edge(torch.cat([d / D.GRID, dist / D.GRID], -1))  # B,6,6,dim

    def message_pass(self, h, e, pad):
        """Returns updated node states [B,N,d] (N=6 objects, no query node here)."""
        B, N, dim = h.shape
        valid = (~pad).float()
        pair_valid = valid[:, :, None] * valid[:, None, :]
        eye = torch.eye(N, device=h.device)[None]
        pair_valid = pair_valid * (1 - eye)
        for r in range(self.rounds):
            hi = h[:, :, None, :].expand(B, N, N, dim)
            hj = h[:, None, :, :].expand(B, N, N, dim)
            m = self.msg[r](torch.cat([hi, hj, e], -1)) * pair_valid[..., None]
            agg = m.sum(2) / pair_valid.sum(2).clamp(min=1)[..., None]
            h = h + self.upd[r](torch.cat([h, agg], -1)) * valid[..., None]
        return h

    def encode_query(self, query):
        emb = self.qemb(query)
        m = (query != D.PAD).float()[..., None]
        return self.qmlp((emb * m).sum(1) / m.sum(1).clamp(min=1))

    def forward(self, objects, query=None, mask_attrs=None, drop_query=None):
        """Solo: query given. mask_attrs/drop_query: [B] bool or None.
        Returns (logits, node_states, pooled)."""
        h, pad = self.embed_nodes(objects, mask_attrs)
        if query is not None:
            qv = self.encode_query(query)[:, None, :]
            if drop_query is not None:
                qv = qv * (~drop_query)[:, None, None].float()
            h = h + qv
        e = self.edge_feats(objects)
        h = self.message_pass(h, e, pad)
        if query is not None:
            qn = self.encode_query(query)[:, None, :]
        else:
            qn = h.mean(1, keepdim=True)
        pooled, _ = self.pool_attn(qn, h, h, key_padding_mask=pad, need_weights=False)
        return self.head(pooled[:, 0]), h, pooled[:, 0]


# ================================================================ C: Interpreter (AR)

class Interpreter(nn.Module):
    """Decoder-only transformer. Solo: [scene_seq][query][<q>] -> answer at last pos.
    Union: [BOS][query][<q>]."""

    def __init__(self, dim=DIM, depth=4, heads=4, max_len=D.SEQ_LEN):
        super().__init__()
        self.emb = nn.Embedding(D.VOCAB_SIZE, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(D.ANSWERS))

    def trunk(self, seq):
        L = seq.shape[1]
        x = self.emb(seq) + self.pos[:, :L]
        cm = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        pad = seq == D.PAD
        for b in self.blocks:
            x = b(x, attn_mask=cm, key_padding_mask=pad)
        return self.ln(x)

    def forward(self, seq):
        return self.head(self.trunk(seq)[:, -1])


def build_solo_seq(scene_seq, query, drop_scene_mask=None):
    """[scene_seq][query][<q>]; optionally replace scene with just BOS per-sample."""
    B = scene_seq.shape[0]
    qcol = torch.full((B, 1), D.QTOK, dtype=scene_seq.dtype, device=scene_seq.device)
    seq = torch.cat([scene_seq, query, qcol], 1)
    if drop_scene_mask is not None and drop_scene_mask.any():
        bos_only = torch.full_like(scene_seq, D.PAD)
        bos_only[:, 0] = D.BOS
        seq2 = torch.cat([bos_only, query, qcol], 1)
        seq = torch.where(drop_scene_mask[:, None], seq2, seq)
    return seq


def build_query_seq(query):
    """Union-mode input for C: [BOS][query][<q>] padded to same layout as solo."""
    B = query.shape[0]
    bos = torch.full((B, D.SCENE_LEN), D.PAD, dtype=query.dtype, device=query.device)
    bos[:, 0] = D.BOS
    qcol = torch.full((B, 1), D.QTOK, dtype=query.dtype, device=query.device)
    return torch.cat([bos, query, qcol], 1)


# ================================================================ B8: Monolith

class Monolith(nn.Module):
    """Bidirectional transformer over [img patches][scene tokens][query][<q>], full input."""

    def __init__(self, dim=256, depth=6, heads=8):
        super().__init__()
        self.patch = nn.Linear(192, dim)
        self.ppos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        self.emb = nn.Embedding(D.VOCAB_SIZE, dim)
        self.tpos = nn.Parameter(torch.randn(1, D.SEQ_LEN, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(D.ANSWERS))

    def forward(self, img, scene_seq, query):
        B = img.shape[0]
        p = img.unfold(1, 8, 8).unfold(2, 8, 8).permute(0, 1, 2, 4, 5, 3).reshape(B, 64, 192)
        pt = self.patch(p) + self.ppos
        qcol = torch.full((B, 1), D.QTOK, dtype=scene_seq.dtype, device=scene_seq.device)
        toks = torch.cat([scene_seq, query, qcol], 1)
        tt = self.emb(toks) + self.tpos[:, :toks.shape[1]]
        pad = torch.cat([torch.zeros(B, 64, dtype=torch.bool, device=img.device),
                         toks == D.PAD], 1)
        x = torch.cat([pt, tt], 1)
        for b in self.blocks:
            x = b(x, key_padding_mask=pad)
        return self.head(self.ln(x)[:, -1])


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
