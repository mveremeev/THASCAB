#!/bin/zsh
# Wait for stage 1 to finish, then run stage 2 gate + stage 3 headline.
while pgrep -f stage1_solos.py >/dev/null; do sleep 60; done
if ! grep -q "STAGE 1 COMPLETE" runs/stage1.log; then
  echo "CHAIN ABORTED: stage 1 did not complete cleanly" >&2
  exit 1
fi
echo "=== stage 1 complete; running stage 2 gate ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/stage2_gate.py 2>&1 | tee runs/stage2_gate.log
echo "=== running stage 3 headline ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/stage3_headline.py 2>&1 | tee runs/stage3.log
echo "=== CHAIN DONE ==="
