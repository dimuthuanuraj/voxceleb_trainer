#!/usr/bin/env bash
set -eo pipefail
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer
setsid nohup python -u tools/eer_power.py --all --bootstrap 1000 \
    > data/_qc/power.log 2>&1 < /dev/null &
echo "launched pid $!"
