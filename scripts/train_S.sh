#!/usr/bin/env bash
# Pretrain CFD2vec-S on the urban pool; re-running the same command resumes from runs/<name>/last.pt.
#   bash scripts/train_S.sh                 masked field modelling  -> runs/S_urban_v2_masked
#   OBJECTIVE=supervised bash scripts/train_S.sh                   -> runs/S_urban_v2_supervised
#   DEVICE=mps bash scripts/train_S.sh      force a device (default auto: cuda, then mps, then cpu)
#   bash scripts/train_S.sh --grad-checkpoint on     lower memory, slower; safe to switch when resuming
# Monitor: runs/<name>/status.json (live), train.log, log.jsonl. Stop with Ctrl-C (checkpoint is saved).
set -euo pipefail
cd "$(dirname "$0")/.."
OBJECTIVE="${OBJECTIVE:-masked}"
NAME="${NAME:-S_urban_v2_${OBJECTIVE}}"
DEVICE="${DEVICE:-auto}"
CONFIG="${CONFIG:-configs/pretrain_S_urban.yaml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1        # any operator without an accelerator kernel falls back to CPU
mkdir -p "runs/$NAME"
RESUME=""; [ -f "runs/$NAME/last.pt" ] && RESUME="--resume"
KEEP_AWAKE=""; command -v caffeinate >/dev/null 2>&1 && KEEP_AWAKE="caffeinate -i"   # no idle sleep during the run
$KEEP_AWAKE python -m cfd2vec.cli pretrain --config "$CONFIG" --out "runs/$NAME" --objective "$OBJECTIVE" \
       --device "$DEVICE" $RESUME "$@" 2>&1 | tee -a "runs/$NAME/console.log"
