# Cluster ticket — NVIDIA driver/library mismatch on compute-node-2 and compute-node-4

Raised: 2026-09-12T08:11:20Z
Raised by: SL_SPV speaker-verification project (Dimuthu Anuraj)
Account has no sudo on the affected nodes — this needs an administrator.

## Impact

Three of five compute nodes cannot run CUDA. Usable capacity is 2x Tesla T4
(15 GB). The project's highest-priority experiment needs a 30 GB card and
therefore cannot be placed at all.

Approximately **760 of 1,100 queued GPU-hours are blocked on this ticket**.

## Symptom

```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.178
```

## Evidence, probed at 2026-09-12T08:11:20Z

| Node | Reachable | nvidia-smi | Kernel module (NVRM) | Userspace libnvidia-ml | Uptime |
|---|---|---|---|---|---|
| `compute-node-1` | yes | OK | `570.211.01` | `570.211.01` | up 1 day, 20 hours, 11 minutes |
| `compute-node-2` | yes | **FAIL** | `580.173.02` | `580.178.04` | up 1 day, 20 hours, 11 minutes |
| `compute-node-3` | **NO** | — | — | — | — |
| `compute-node-4` | yes | **FAIL** | `580.173.02` | `580.178.04` | up 1 day, 14 hours, 31 minutes |
| `compute-node-5` | yes | **FAIL** | `-` | `-` | up 1 day, 20 hours, 11 minutes |

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
