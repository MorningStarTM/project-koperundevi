#!/bin/bash
# Phase 4: restart a crashed/killed run, resuming from the last checkpoint.
# Usage: ./supervise.sh <same args you'd pass to train_gpt.py after --resume>
set -u

while true; do
  ./run_train.sh --resume "$@"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "training finished cleanly"
    break
  fi
  echo "training exited with code $code, restarting in 10s ..."
  sleep 10
done
