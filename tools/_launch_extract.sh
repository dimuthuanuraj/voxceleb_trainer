#!/usr/bin/env bash
# Detached launcher for the full embedding extraction (run via tools/noderun.sh 3).
set -eo pipefail
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer
mkdir -p data/_qc
setsid nohup python -u tools/extract_embeddings.py \
    --all --per-speaker 60 --max-utts 25000 --batch 96 --overwrite \
    > data/_qc/extract.log 2>&1 < /dev/null &
echo "launched pid $!"
