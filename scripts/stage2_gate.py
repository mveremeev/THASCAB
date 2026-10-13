"""Stage 2 gate (fair, uncontended): scratch union vs B3 solo on curriculum,
equal epoch budget."""

import sys

sys.path.insert(0, ".")

from thascab import models as M
from thascab import train as T
from thascab import union_train as UT

splits = T.load_splits("data/curriculum", names=("train", "iid"))
print("[gate] B3 solo on curriculum, 15 epochs")
T.set_seed(0)
b3 = T.train_loop(M.Interpreter(), T.loss_interpreter, splits, "cur_b3_15ep",
                  epochs=15, seed=0, need_images=False, eval_every=5)
print("[gate] scratch union on curriculum, 15 epochs")
_, uf, _ = UT.train_union(splits, "cur_union_15ep", seed=0, regime="scratch",
                          epochs=15, eval_every=5)
b3a, ua = b3["iid"]["overall"], uf["iid"]["overall"]
print(f"\nSTAGE2 GATE: B3={b3a:.3f} union={ua:.3f} -> "
      f"{'PASS' if ua >= b3a - 0.02 else 'MARGINAL/FAIL (review)'}")
