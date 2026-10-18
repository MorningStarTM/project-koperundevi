#!/bin/bash
# Picks the right launch mode based on what hardware is actually there:
#   0 GPUs -> plain CPU training
#   1 GPU  -> plain single-GPU training (no DDP overhead for one device)
#   2+ GPUs -> torchrun, one process per GPU, DDP (see gpt/trainer.py is_ddp())
set -eu

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

if [ "$GPU_COUNT" -ge 2 ]; then
  echo "Detected $GPU_COUNT GPUs -> DDP training"
  exec torchrun --standalone --nproc_per_node="$GPU_COUNT" train_gpt.py "$@"
elif [ "$GPU_COUNT" -eq 1 ]; then
  echo "Detected 1 GPU -> single-GPU training"
  exec python3 train_gpt.py "$@"
else
  echo "No GPU detected -> CPU training"
  exec python3 train_gpt.py "$@"
fi
