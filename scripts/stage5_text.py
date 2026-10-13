"""Stage 5: B6 text-bridge (discrete channel), bandwidth-matched, 5 seeds.
Gumbel-softmax annealed 5.0 -> 0.5; frozen cores + cache (same as B4 for a
clean latent-vs-symbolic comparison)."""

import sys
import time

sys.path.insert(0, ".")

from thascab import train as T
from thascab import union_train as UT

SEEDS = [0, 1, 2, 3, 4]
splits = T.load_splits()
t00 = time.time()

for seed in SEEDS:
    print(f"[B6 text bridge, seed {seed}]", flush=True)
    UT.train_union(splits, f"b6_s{seed}", seed=seed, regime="frozen",
                   epochs=25, lr=1e-3, use_cache=True, text_bridge=True)

print(f"STAGE 5 COMPLETE in {time.time()-t00:.0f}s", flush=True)
