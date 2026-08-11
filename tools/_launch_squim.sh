#!/usr/bin/env bash
set -eo pipefail
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer
setsid nohup python -u tools/signal_quality.py --all --sample 1200 --squim --squim-n 500 \
    > data/_qc/signal_squim.log 2>&1 < /dev/null &
echo "launched pid $!"
