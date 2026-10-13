#!/bin/zsh
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
run() { python3 scripts/run_union_cfg.py "$@" || exit 1 }
echo "[A: b4+module_dropout 0.3 (anti lazy-fusion)]"
run --name diag_drop_s0 --module_dropout 0.3 --epochs 30 --damage
echo "[B: b4.5 finetune from b4_s0]"
run --name b45_s0 --regime finetune --epochs 12 --lr 3e-4 --init_from runs/b4_s0/model.pt --damage
echo "[C: R=5]"
run --name diag_r5_s0 --R 5 --epochs 25 --eval_every 25 --damage
echo "[D: K=16]"
run --name diag_k16_s0 --K 16 --epochs 25 --eval_every 25 --damage
echo "DIAG2 COMPLETE"
