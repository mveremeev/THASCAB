"""MiniCLEVR-R: procedural relational visual-reasoning dataset (THASCAB Stage 0).

Scene: 3-6 objects on a 6x6 grid, rendered both as a 64x64 RGB image and a
symbolic object list. Queries: templated relational questions, 1-3 hops.
Answer: single class out of 17.

Leakage control: every (scene, query) pair is canonically hashed; scenes are
globally unique across splits (a scene hash appears in exactly one split), and
pairs are deduplicated by pair hash.

Splits:
  train       3-5 objects, no held-out attr combos, no held-out templates
  iid         same distribution as train, disjoint scenes (by scene hash)
  comp_attr   scenes contain >=1 held-out (color,shape,size) combo
  comp_tmpl   standard scenes, held-out query templates only
  extrap      6-object scenes (train has 3-5)
"""

import hashlib
import json
import os

import numpy as np

GRID = 6
IMG = 64
MAX_OBJ = 6

SHAPES = ["circle", "square", "triangle"]
COLORS = ["red", "green", "blue"]
SIZES = ["small", "large"]

COLOR_RGB = {"red": (230, 60, 50), "green": (60, 200, 80), "blue": (60, 110, 230)}
BG = 26

# Held out from ALL training scenes; only in comp_attr scenes (color, shape, size)
HELDOUT_COMBOS = {
    ("red", "triangle", "large"),
    ("blue", "circle", "small"),
    ("green", "square", "large"),
}

# Held-out template variants (family, key) -- never generated for train/iid/extrap
HELDOUT_TEMPLATES = {
    ("rel_attr_1", "above|size"),   # "what size is the object above the X"
    ("count_rel", "left"),          # "how many objects are left of the X"
    ("same_attr", "size"),          # "are X and Y the same size"
}

FAMILIES = ["pos_attr", "rel_attr_1", "rel_attr_2", "rel_attr_3",
            "count_rel", "count_attr", "count_cmp", "same_attr"]
FAM2ID = {f: i for i, f in enumerate(FAMILIES)}

ANSWERS = [str(i) for i in range(7)] + SHAPES + COLORS + SIZES + ["yes", "no"]
ANS2ID = {a: i for i, a in enumerate(ANSWERS)}

_WORDS = (
    ["<pad>", "<bos>", "<sep>", "<q>", "obj"]
    + [str(i) for i in range(7)]
    + SHAPES + COLORS + SIZES
    + ["what", "is", "the", "object", "at", "left", "right", "above", "below",
       "of", "nearest", "to", "how", "many", "objects", "are", "there", "more",
       "than", "things", "thing", "and", "same", "do", "have", "shape", "color",
       "size"]
)
VOCAB = {w: i for i, w in enumerate(_WORDS)}
VOCAB_SIZE = len(_WORDS)
PAD, BOS, SEP, QTOK = VOCAB["<pad>"], VOCAB["<bos>"], VOCAB["<sep>"], VOCAB["<q>"]

Q_LEN = 24          # padded query length
SCENE_LEN = 1 + MAX_OBJ * 6 + 1   # bos + 6 tokens per object + sep
SEQ_LEN = SCENE_LEN + Q_LEN + 1   # + <q> answer-position token

RELS = ["left", "right", "above", "below", "nearest"]
REL_WORDS = {"left": ["left", "of"], "right": ["right", "of"],
             "above": ["above"], "below": ["below"], "nearest": ["nearest", "to"]}


# ---------------------------------------------------------------- scenes

def sample_scene(rng, n_obj, require_heldout=False, forbid_heldout=True):
    """Scene = list of (x, y, shape, color, size) tuples, unique cells."""
    while True:
        cells = rng.choice(GRID * GRID, size=n_obj, replace=False)
        objs = []
        for c in cells:
            x, y = int(c % GRID), int(c // GRID)
            objs.append((x, y, SHAPES[rng.integers(3)], COLORS[rng.integers(3)],
                         SIZES[rng.integers(2)]))
        combos = {(o[3], o[2], o[4]) for o in objs}
        has_ho = bool(combos & HELDOUT_COMBOS)
        if require_heldout and not has_ho:
            continue
        if forbid_heldout and has_ho:
            continue
        return sorted(objs)


def scene_hash(objs):
    s = ";".join(",".join(map(str, o)) for o in sorted(objs))
    return hashlib.sha256(s.encode()).hexdigest()


def pair_hash(objs, qtokens):
    s = scene_hash(objs) + "|" + ",".join(map(str, qtokens))
    return hashlib.sha256(s.encode()).hexdigest()


def render(objs):
    """64x64 RGB uint8. Deterministic jitter seeded by scene hash."""
    img = np.full((IMG, IMG, 3), BG, dtype=np.uint8)
    jr = np.random.default_rng(int(scene_hash(objs)[:8], 16))
    cell = IMG / GRID
    Y, X = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    for (x, y, shape, color, size) in objs:
        cx = (x + 0.5) * cell + jr.uniform(-1.5, 1.5)
        cy = (y + 0.5) * cell + jr.uniform(-1.5, 1.5)
        r = 4.5 if size == "large" else 2.6
        if shape == "circle":
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        elif shape == "square":
            mask = np.maximum(np.abs(X - cx), np.abs(Y - cy)) <= r * 0.92
        else:  # triangle (pointing up)
            x1, y1 = cx, cy - r * 1.1
            x2, y2 = cx - r, cy + r * 0.8
            x3, y3 = cx + r, cy + r * 0.8
            d1 = (X - x2) * (y1 - y2) - (Y - y2) * (x1 - x2)
            d2 = (X - x3) * (y2 - y3) - (Y - y3) * (x2 - x3)
            d3 = (X - x1) * (y3 - y1) - (Y - y1) * (x3 - x1)
            mask = ((d1 <= 0) & (d2 <= 0) & (d3 <= 0)) | ((d1 >= 0) & (d2 >= 0) & (d3 >= 0))
        img[mask] = COLOR_RGB[color]
    return img


# ---------------------------------------------------------------- referring expressions

def _matches(o, desc):
    for k, v in desc.items():
        idx = {"shape": 2, "color": 3, "size": 4}[k]
        if o[idx] != v:
            return False
    return True


def unique_ref(rng, objs, target, avoid=None):
    """A descriptor uniquely identifying `target` in scene, or None.
    avoid: attribute name that must NOT appear in the descriptor."""
    x, y, shape, color, size = target
    cands = [{"shape": shape}, {"color": color},
             {"color": color, "shape": shape},
             {"size": size, "shape": shape}, {"size": size, "color": color},
             {"size": size, "color": color, "shape": shape}]
    if avoid is not None:
        cands = [c for c in cands if avoid not in c]
    rng.shuffle(cands)
    for d in cands:
        if sum(_matches(o, d) for o in objs) == 1:
            return d
    return None


def ref_words(desc):
    w = ["the"]
    if "size" in desc:
        w.append(desc["size"])
    if "color" in desc:
        w.append(desc["color"])
    w.append(desc.get("shape", "thing"))
    return w


# ---------------------------------------------------------------- relations

def rel_set(objs, anchor, rel):
    ax, ay = anchor[0], anchor[1]
    if rel == "left":
        return [o for o in objs if o[0] < ax]
    if rel == "right":
        return [o for o in objs if o[0] > ax]
    if rel == "above":
        return [o for o in objs if o[1] < ay]
    if rel == "below":
        return [o for o in objs if o[1] > ay]
    # nearest: unique argmin distance, tie -> []
    others = [o for o in objs if o != anchor]
    d = [(o[0] - ax) ** 2 + (o[1] - ay) ** 2 for o in others]
    m = min(d)
    return [others[d.index(m)]] if d.count(m) == 1 else []


def unique_rel_target(objs, anchor, rel):
    s = rel_set(objs, anchor, rel)
    return s[0] if len(s) == 1 else None


# ---------------------------------------------------------------- query generation

def _tokens(words):
    return [VOCAB[w] for w in words]


def gen_query(rng, objs, family, allow_heldout_tmpl=False, force_heldout_tmpl=False):
    """Returns (qtokens, answer_str, tmpl_key) or None if invalid draw."""
    Q = {"shape": 2, "color": 3, "size": 4}

    def is_ho(fam, key):
        return (fam, key) in HELDOUT_TEMPLATES

    if family == "pos_attr":
        if force_heldout_tmpl:
            return None
        t = objs[rng.integers(len(objs))]
        attr = ["shape", "color"][rng.integers(2)]
        words = ["what", attr, "is", "the", "object", "at", str(t[0]), str(t[1])]
        return _tokens(words), t[Q[attr]], "pos"

    if family in ("rel_attr_1", "rel_attr_2", "rel_attr_3"):
        hops = {"rel_attr_1": 1, "rel_attr_2": 2, "rel_attr_3": 3}[family]
        if force_heldout_tmpl and hops != 1:
            return None
        base = objs[rng.integers(len(objs))]
        desc = unique_ref(rng, objs, base)
        if desc is None:
            return None
        cur, rels = base, []
        for h in range(hops):
            # comp_tmpl: force the exact held-out variant (above|size, 1-hop)
            rel = "above" if (force_heldout_tmpl and h == 0) else RELS[rng.integers(len(RELS))]
            nxt = unique_rel_target(objs, cur, rel)
            if nxt is None:
                return None
            rels.append(rel)
            cur = nxt
        if cur == base:
            return None
        attr = "size" if force_heldout_tmpl else ["shape", "color", "size"][rng.integers(3)]
        key = f"{rels[0]}|{attr}" if hops == 1 else f"{hops}hop"
        if hops == 1 and is_ho(family, key) and not (allow_heldout_tmpl or force_heldout_tmpl):
            return None
        # innermost relation is the first walk step from the referenced object:
        # walk [r1, r2] => "the object r2 (the object r1 (the ref))"
        words = ["what", attr, "is"]
        for r in reversed(rels):
            words += ["the", "object"] + REL_WORDS[r]
        words += ref_words(desc)
        return _tokens(words), cur[Q[attr]], key

    if family == "count_rel":
        base = objs[rng.integers(len(objs))]
        desc = unique_ref(rng, objs, base)
        if desc is None:
            return None
        rel = "left" if force_heldout_tmpl else RELS[rng.integers(4)]  # no 'nearest' for counting
        key = rel
        if is_ho(family, key) and not (allow_heldout_tmpl or force_heldout_tmpl):
            return None
        n = len(rel_set(objs, base, rel))
        words = ["how", "many", "objects", "are"] + REL_WORDS[rel] + ref_words(desc)
        return _tokens(words), str(n), key

    if family == "count_attr":
        if force_heldout_tmpl:
            return None
        kind = rng.integers(3)
        if kind == 0:
            v = COLORS[rng.integers(3)]
            n = sum(o[3] == v for o in objs)
            words = ["how", "many", v, "things", "are", "there"]
        elif kind == 1:
            v = SHAPES[rng.integers(3)]
            n = sum(o[2] == v for o in objs)
            words = ["how", "many", v, "things", "are", "there"]
        else:
            v = SIZES[rng.integers(2)]
            n = sum(o[4] == v for o in objs)
            words = ["how", "many", v, "things", "are", "there"]
        return _tokens(words), str(n), "cnt"

    if family == "count_cmp":
        if force_heldout_tmpl:
            return None
        pool = COLORS + SHAPES + SIZES
        a, b = rng.choice(len(pool), 2, replace=False)
        a, b = pool[a], pool[b]

        def cnt(v):
            return sum(v in (o[2], o[3], o[4]) for o in objs)

        ans = "yes" if cnt(a) > cnt(b) else "no"
        words = ["are", "there", "more", a, "things", "than", b, "things"]
        return _tokens(words), ans, "cmp"

    if family == "same_attr":
        if len(objs) < 2:
            return None
        i, j = rng.choice(len(objs), 2, replace=False)
        o1, o2 = objs[i], objs[j]
        attr = "size" if force_heldout_tmpl else ["color", "shape", "size"][rng.integers(3)]
        if is_ho(family, attr) and not (allow_heldout_tmpl or force_heldout_tmpl):
            return None
        # question attr must not be stated in either ref (else trivial)
        d1 = unique_ref(rng, objs, o1, avoid=attr)
        d2 = unique_ref(rng, objs, o2, avoid=attr)
        if d1 is None or d2 is None:
            return None
        ans = "yes" if o1[Q[attr]] == o2[Q[attr]] else "no"
        words = (["do"] + ref_words(d1) + ["and"] + ref_words(d2)
                 + ["have", "the", "same", attr])
        return _tokens(words), ans, attr

    raise ValueError(family)


# ---------------------------------------------------------------- encoding

def encode_query(qtokens):
    q = np.full(Q_LEN, PAD, dtype=np.int16)
    q[:len(qtokens)] = qtokens
    return q


def encode_scene_seq(objs):
    """Symbolic serialization for the Interpreter: bos (obj x y color shape size)* sep."""
    toks = [BOS]
    for (x, y, shape, color, size) in objs:
        toks += [VOCAB["obj"], VOCAB[str(x)], VOCAB[str(y)],
                 VOCAB[color], VOCAB[shape], VOCAB[size]]
    toks.append(SEP)
    out = np.full(SCENE_LEN, PAD, dtype=np.int16)
    out[:len(toks)] = toks
    return out


def encode_objects(objs):
    """Numeric object table for the GNN: [x, y, shape, color, size], pad rows = -1."""
    arr = np.full((MAX_OBJ, 5), -1, dtype=np.int8)
    for i, (x, y, shape, color, size) in enumerate(objs):
        arr[i] = [x, y, SHAPES.index(shape), COLORS.index(color), SIZES.index(size)]
    return arr


# ---------------------------------------------------------------- dataset build

# difficulty mix (calibrated in pilot). family -> sampling weight
DEFAULT_MIX = {"rel_attr_1": 0.22, "rel_attr_2": 0.18, "rel_attr_3": 0.10,
               "count_rel": 0.18, "count_attr": 0.10, "count_cmp": 0.12,
               "same_attr": 0.10}


class _Balancer:
    """Per-family answer balancing via rejection (cap the modal answer)."""

    def __init__(self, cap=0.45):
        self.counts = {}
        self.cap = cap

    def ok(self, fam, ans, rng):
        c = self.counts.setdefault(fam, {})
        tot = sum(c.values()) + 1e-9
        if tot > 200 and c.get(ans, 0) / tot > self.cap:
            return rng.random() < 0.15  # mostly reject over-represented answers
        return True

    def add(self, fam, ans):
        self.counts.setdefault(fam, {})[ans] = self.counts[fam].get(ans, 0) + 1


def build_split(rng, n_pairs, mode, mix, queries_per_scene=3, seen_scenes=None,
                seen_pairs=None, scene_router=None):
    """mode: train | iid | comp_attr | comp_tmpl | extrap.
    scene_router: for train/iid, function(scene_hash)->'train'|'iid' so those two
    splits share a generation stream but are scene-disjoint by hash."""
    fams = list(mix.keys())
    if mode == "comp_tmpl":
        fams = [f for f in fams if any(h[0] == f for h in HELDOUT_TEMPLATES)]
    w = np.array([mix[f] for f in fams])
    w = w / w.sum()
    bal = _Balancer()
    scenes, pairs = [], []  # scenes: (objs, img); pairs: dicts
    want = n_pairs
    while len(pairs) < want:
        if mode == "extrap":
            n = 6
            objs = sample_scene(rng, n, forbid_heldout=True)
        elif mode == "comp_attr":
            n = int(rng.integers(3, 6))
            objs = sample_scene(rng, n, require_heldout=True, forbid_heldout=False)
        else:
            n = int(rng.integers(3, 6))
            objs = sample_scene(rng, n, forbid_heldout=True)
        sh = scene_hash(objs)
        if sh in seen_scenes:
            continue
        if scene_router is not None:
            if scene_router(sh) != mode:
                continue
        seen_scenes.add(sh)
        got = 0
        s_idx = None
        for _ in range(queries_per_scene * 4):
            if got >= queries_per_scene or len(pairs) >= want:
                break
            # stick with the drawn family for many attempts so the realized
            # mix tracks the intended mix (easy families must not win by
            # rejection attrition)
            fam = fams[int(rng.choice(len(fams), p=w))]
            out = None
            for _ in range(40):
                cand = gen_query(rng, objs, fam,
                                 force_heldout_tmpl=(mode == "comp_tmpl"))
                if cand is None:
                    continue
                # dup / answer-balance rejection retries the SAME family so the
                # realized mix tracks the intended mix
                if pair_hash(objs, cand[0]) in seen_pairs:
                    continue
                if not bal.ok(fam, cand[1], rng):
                    continue
                out = cand
                break
            if out is None:
                continue
            qtok, ans, key = out
            ph = pair_hash(objs, qtok)
            seen_pairs.add(ph)
            bal.add(fam, ans)
            if s_idx is None:
                s_idx = len(scenes)
                scenes.append(objs)
            pairs.append({"scene": s_idx, "q": qtok, "ans": ans, "fam": fam})
            got += 1
    return scenes, pairs


def materialize(scenes, pairs, path):
    imgs = np.stack([render(o) for o in scenes]) if scenes else np.zeros((0, IMG, IMG, 3), np.uint8)
    obj_tab = np.stack([encode_objects(o) for o in scenes])
    scene_seq = np.stack([encode_scene_seq(o) for o in scenes])
    sidx = np.array([p["scene"] for p in pairs], np.int32)
    qtok = np.stack([encode_query(p["q"]) for p in pairs])
    ans = np.array([ANS2ID[p["ans"]] for p in pairs], np.int16)
    fam = np.array([FAM2ID[p["fam"]] for p in pairs], np.int8)
    np.savez_compressed(path, images=imgs, objects=obj_tab, scene_seq=scene_seq,
                        scene_idx=sidx, query=qtok, answer=ans, family=fam)
    return {"scenes": len(scenes), "pairs": len(pairs)}


def build_all(out_dir, sizes=None, mix=None, seed=0, curriculum=False):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    mix = mix or DEFAULT_MIX
    if curriculum:
        mix = {"pos_attr": 1.0}
        sizes = sizes or {"train": 15000, "iid": 2000}
    sizes = sizes or {"train": 40000, "iid": 6000, "comp_attr": 4000,
                      "comp_tmpl": 4000, "extrap": 6000}
    seen_scenes, seen_pairs = set(), set()

    def router(sh):
        return "train" if int(sh[:8], 16) % 100 < 87 else "iid"

    stats = {}
    for split in sizes:
        sc, pr = build_split(rng, sizes[split], split, mix,
                             queries_per_scene=3 if split == "train" else 2,
                             seen_scenes=seen_scenes, seen_pairs=seen_pairs,
                             scene_router=router if split in ("train", "iid") and not curriculum else None)
        stats[split] = materialize(sc, pr, os.path.join(out_dir, f"{split}.npz"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"sizes": {k: v for k, v in stats.items()}, "mix": mix,
                   "seed": seed, "vocab_size": VOCAB_SIZE,
                   "n_answers": len(ANSWERS), "curriculum": curriculum}, f, indent=2)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/full")
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(build_all(args.out, seed=args.seed, curriculum=args.curriculum))
