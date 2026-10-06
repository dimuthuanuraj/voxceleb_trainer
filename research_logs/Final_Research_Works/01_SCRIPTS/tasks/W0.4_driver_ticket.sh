#!/usr/bin/env bash
# W0.4 -- Escalate the NVIDIA driver mismatch to the cluster administrator.
#
# This task cannot fix the problem: `sudo -n` fails for this account on both
# affected nodes. What it does is produce an exact, self-contained ticket with
# the evidence and the commands, so the administrator does not have to
# rediscover anything -- and re-probe so the exit test flips automatically once
# the nodes come back.
#
# Exit test: 3+ GPUs answer nvidia-smi across the cluster.

. "$(dirname "$0")/../lib/common.sh"

TICKET="$RESULTS/W0.4/DRIVER_TICKET.md"
mkdir -p "$(dirname "$TICKET")"

step "Probing every node for the evidence"
{
  echo "# Cluster ticket — NVIDIA driver/library mismatch on compute-node-2 and compute-node-4"
  echo
  echo "Raised: $(_ts)"
  echo "Raised by: SL_SPV speaker-verification project (Dimuthu Anuraj)"
  echo "Account has no sudo on the affected nodes — this needs an administrator."
  echo
  echo "## Impact"
  echo
  echo "Three of five compute nodes cannot run CUDA. Usable capacity is 2x Tesla T4"
  echo "(15 GB). The project's highest-priority experiment needs a 30 GB card and"
  echo "therefore cannot be placed at all."
  echo
  echo "Approximately **760 of 1,100 queued GPU-hours are blocked on this ticket**."
  echo
  echo "## Symptom"
  echo
  echo '```'
  echo "Failed to initialize NVML: Driver/library version mismatch"
  echo "NVML library version: 580.178"
  echo '```'
  echo
  echo "## Evidence, probed at $(_ts)"
  echo
  echo '| Node | Reachable | nvidia-smi | Kernel module (NVRM) | Userspace libnvidia-ml | Uptime |'
  echo '|---|---|---|---|---|---|'
} > "$TICKET"

for n in compute-node-1 compute-node-2 compute-node-3 compute-node-4 compute-node-5; do
    if ! timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=6 "$n" true 2>/dev/null; then
        echo "| \`$n\` | **NO** | — | — | — | — |" >> "$TICKET"
        log "  $n unreachable"
        continue
    fi
    smi=$(timeout 15 ssh -o BatchMode=yes "$n" 'nvidia-smi -L 2>&1 | head -2 | tr "\n" " "' 2>/dev/null || echo "?")
    krn=$(timeout 10 ssh -o BatchMode=yes "$n" "awk '{print \$8}' /proc/driver/nvidia/version 2>/dev/null" 2>/dev/null || echo "-")
    usr=$(timeout 10 ssh -o BatchMode=yes "$n" "ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.* 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1" 2>/dev/null || echo "-")
    upt=$(timeout 10 ssh -o BatchMode=yes "$n" "uptime -p" 2>/dev/null || echo "-")
    ok=$(echo "$smi" | grep -q "GPU 0" && echo "OK" || echo "**FAIL**")
    echo "| \`$n\` | yes | $ok | \`${krn:--}\` | \`${usr:--}\` | ${upt:--} |" >> "$TICKET"
    log "  $n: smi=$ok krn=$krn usr=$usr"
done

cat >> "$TICKET" <<'MD'

## Diagnosis

The NVIDIA **kernel module** in memory is `580.173.02`; the **userspace** library
`dpkg` installed is `580.178.04`. NVML refuses to initialise across that skew.
The package was upgraded underneath a running module.

Both nodes have been rebooted *since* the upgrade (uptimes above) and still
mismatch, which points at one of:

* `compute-node-2` still carries the superseded 535 branch —
  `nvidia-driver-535 535.309.01` and `libnvidia-compute-535` are both installed
  alongside 580, left from the migration of 2026-08-15. Module selection may be
  picking the wrong branch at boot.
* a stale initramfs pinning the older module on `compute-node-4`.

## Requested fix

```bash
# On compute-node-2 and compute-node-4, as root:
nvidia-smi                       # confirm the failure first

# 1. Reload the module stack (no reboot needed if nothing holds the GPUs):
systemctl isolate multi-user.target
rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
modprobe nvidia
nvidia-smi                       # must now list the cards

# 2. If that fails, the initramfs is stale:
update-initramfs -u -k all && reboot

# 3. compute-node-2 only — remove the superseded branch:
apt-get purge -y 'nvidia-*-535' 'libnvidia-*-535'
apt-mark hold nvidia-driver-580  # stop it moving under a running module again
update-initramfs -u -k all && reboot
```

## Also requested

`compute-node-3` (10.222.1.120, 2x NVIDIA A10 23 GB) has not routed since
**2026-08-15** — `No route to host`. Recovering it would add two 23 GB cards,
which are large enough for the blocked 30 GB experiment if node-4 cannot be
restored.

## Verification

The project re-probes automatically. After the fix, from the head node:

```bash
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/research_logs/Final_Research_Works/01_SCRIPTS
./preflight.sh
python3 runner.py --status
```

`W0.4` flips to PASS once three or more GPUs answer `nvidia-smi`.
MD

step "Current state"
python3 - <<'PY'
import subprocess, sys, os
sys.path.insert(0, os.path.join(os.environ["SCRIPTS"]))
from runner import probe_gpu_count
n, detail = probe_gpu_count()
print(f"  usable GPUs now: {n}")
print(f"  {detail}")
if n >= 3:
    print("\n  W0.4 already satisfied -- the cluster has recovered.")
else:
    print(f"\n  W0.4 NOT satisfied: {n} GPU(s) usable, need 3+.")
    print("  Send the ticket. This task passes automatically once the nodes return.")
PY

log "ticket written: $TICKET"
echo
echo "  Send: $TICKET"
