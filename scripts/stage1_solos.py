"""Stage 1: train all solo baselines (B1 frozen+ft, B2, B3) for 5 seeds."""

import sys
import time

sys.path.insert(0, ".")

import torch

from thascab import models as M
from thascab import train as T

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 20

splits = T.load_splits()
t00 = time.time()

for seed in SEEDS:
    print(f"===== seed {seed} =====", flush=True)

    print(f"[B3 interpreter solo, seed {seed}]", flush=True)
    T.set_seed(seed)
    T.train_loop(M.Interpreter(), T.loss_interpreter, splits, f"b3_s{seed}",
                 epochs=EPOCHS, seed=seed, need_images=False)

    print(f"[B2 reasoner solo, seed {seed}]", flush=True)
    T.set_seed(seed)
    T.train_loop(M.Reasoner(), T.loss_reasoner, splits, f"b2_s{seed}",
                 epochs=EPOCHS, seed=seed, need_images=False)

    print(f"[JEPA pretrain, seed {seed}]", flush=True)
    enc = T.jepa_pretrain(splits, f"jepa_s{seed}", epochs=30, seed=seed)

    print(f"[B1 perceiver solo (frozen enc), seed {seed}]", flush=True)
    T.train_perceiver_solo(splits, enc, f"b1frozen_s{seed}", finetune=False,
                           epochs=EPOCHS, seed=seed)

    print(f"[B1 perceiver solo (finetuned enc), seed {seed}]", flush=True)
    enc2 = M.ViTEncoder()
    enc2.load_state_dict(torch.load(f"runs/jepa_s{seed}/encoder.pt"))
    T.train_perceiver_solo(splits, enc2, f"b1ft_s{seed}", finetune=True,
                           epochs=EPOCHS, seed=seed)

    print(f"seed {seed} done at {time.time()-t00:.0f}s total", flush=True)

print(f"STAGE 1 COMPLETE in {time.time()-t00:.0f}s", flush=True)
