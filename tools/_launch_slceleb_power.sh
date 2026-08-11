#!/usr/bin/env bash
set -eo pipefail
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer
setsid nohup python -u tools/eer_power.py --dataset data/slceleb2026_sinhala \
    --bootstrap 1000 > data/_qc/power_slceleb.log 2>&1 < /dev/null &
echo "launched pid $!"
