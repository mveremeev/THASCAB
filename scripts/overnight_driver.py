"""Overnight driver: waits for the diagnostic ladder, picks the best union
config, then runs the entire remaining experiment queue unattended.

Order (highest scientific priority first):
  1. headline fleet: B4 x5 (cached), B4.5 x5 (uncapped bs64), B5 x5
  2. nulls: B7 ensemble, B8 monolith sweep + fleet
  3. B6 text bridge x5
  4. K-sweep / R-sweep / b4drop (3 seeds, 15 ep)
  5. aggregate + figures

Every run: own process, skip if runs/<name>/result.json exists, 3h timeout.
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CAPPED = {"PYTORCH_ENABLE_MPS_FALLBACK": "1",
          "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0",
          "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.8"}
UNCAPPED = {"PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}


def log(msg):
    print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cfg(name, extra, env, timeout=3 * 3600):
    if os.path.exists(f"runs/{name}/result.json"):
        log(f"skip {name} (done)")
        return True
    cmd = ["python3", "scripts/run_union_cfg.py", "--name", name] + extra
    log("RUN " + " ".join(cmd))
    e = dict(os.environ)
    e.update(env)
    try:
        r = subprocess.run(cmd, env=e, timeout=timeout)
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {name}")
        ok = False
    if not ok:
        log(f"FAILED {name} (continuing)")
    return ok


# ---- phase 0: wait for the running diagnostic ladder ----
log("waiting for diag2c ladder to finish...")
while subprocess.run(["pgrep", "-f", "run_union_cfg.py|stage3_diag2c"],
                     capture_output=True).returncode == 0:
    time.sleep(120)
log("ladder done")

# ---- phase 1: pick union config ----
cands = {(8, 3): "b4_s0", (8, 5): "diag_r5_s0", (16, 3): "diag_k16_s0"}
best_cfg, best_acc = (8, 3), -1.0
for (K, R), name in cands.items():
    p = f"runs/{name}/result.json"
    if os.path.exists(p):
        acc = json.load(open(p))["final"]["iid"]["overall"]
        log(f"cand K={K} R={R}: iid={acc:.3f}")
        if acc > best_acc + 0.005:  # require a real margin to deviate
            best_acc, best_cfg = acc, (K, R)
K, R = best_cfg
log(f"chosen config: K={K} R={R}")
KR = ["--K", str(K), "--R", str(R)]
json.dump({"K": K, "R": R}, open("runs/union_config.json", "w"))

# if config != (8,3), the existing b4_s0 was trained with the old config:
# retrain under the fleet name so the fleet is homogeneous.
if best_cfg != (8, 3):
    for f in ("result.json", "model.pt", "damage.json"):
        p = f"runs/b4_s0/{f}"
        if os.path.exists(p):
            os.rename(p, p + ".k8r3")

# ---- phase 2: headline fleet ----
run_cfg("b4_s0", KR + ["--epochs", "25", "--damage", "--ckpts"], CAPPED)
for s in (1, 2, 3, 4):
    run_cfg(f"b4_s{s}", KR + ["--seed", str(s), "--epochs", "25", "--damage"],
            CAPPED)
for s in (0, 1, 2, 3, 4):
    if os.path.exists(f"runs/b4_s{s}/model.pt"):
        run_cfg(f"b45_s{s}", KR + ["--seed", str(s), "--regime", "finetune",
                                   "--epochs", "10", "--lr", "3e-4", "--bs", "64",
                                   "--init_from", f"runs/b4_s{s}/model.pt",
                                   "--damage"], UNCAPPED)
for s in (0, 1, 2, 3, 4):
    run_cfg(f"b5_s{s}", KR + ["--seed", str(s), "--regime", "scratch",
                              "--epochs", "15", "--lr", "1e-3", "--bs", "64",
                              "--damage"], UNCAPPED)

# ---- phase 3: nulls (ensemble + monolith), own script ----
if not os.path.exists("runs/b8_s4/result.json"):
    log("RUN stage4_nulls.py")
    e = dict(os.environ)
    e.update(CAPPED)
    try:
        subprocess.run(["python3", "scripts/stage4_nulls.py"], env=e,
                       timeout=6 * 3600)
    except subprocess.TimeoutExpired:
        log("TIMEOUT stage4_nulls")

# ---- phase 4: B6 text bridge ----
for s in (0, 1, 2, 3, 4):
    run_cfg(f"b6_s{s}", KR + ["--seed", str(s), "--epochs", "25",
                              "--text_bridge", "--damage"], CAPPED)

# ---- phase 5: K/R sweeps + dropout variant (3 seeds, 15 ep) ----
for k in (1, 2, 4, 16):
    for s in (0, 1, 2):
        run_cfg(f"ksweep_k{k}_s{s}", ["--K", str(k), "--R", "3", "--seed", str(s),
                                      "--epochs", "15", "--eval_every", "15"],
                CAPPED, timeout=2 * 3600)
for r in (1, 2, 5):
    for s in (0, 1, 2):
        run_cfg(f"rsweep_r{r}_s{s}", ["--K", "8", "--R", str(r), "--seed", str(s),
                                      "--epochs", "15", "--eval_every", "15"],
                CAPPED, timeout=2 * 3600)
for s in (1, 2):
    run_cfg(f"b4drop_s{s}", ["--K", "8", "--R", "3", "--seed", str(s),
                             "--epochs", "25", "--module_dropout", "0.3",
                             "--damage", "--eval_every", "25"], CAPPED,
            timeout=2 * 3600)

# ---- phase 6: aggregate ----
log("aggregating")
subprocess.run(["python3", "scripts/aggregate.py"])
log("OVERNIGHT DRIVER COMPLETE")
