#!/usr/bin/env bash
set -eo pipefail
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer
setsid nohup python -u tools/extract_embeddings.py \
    --dataset data/slceleb2026_sinhala --per-speaker 60 --max-utts 25000 \
    --batch 96 --overwrite > data/_qc/extract_slceleb.log 2>&1 < /dev/null &
echo "launched pid $!"
