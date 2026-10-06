---
title: "Final Research Works — Blockers and Risks"
author: "Dimuthu Anuraj"
date: "2026-09-12"
---

# 1. Blockers, by severity

## B-1 — CRITICAL: three of five compute nodes cannot run CUDA

**Measured 2026-09-12.** `compute-node-2` and `compute-node-4` return
`Failed to initialize NVML: Driver/library version mismatch`; `compute-node-3`
does not route.

| | Kernel module | Userspace | Verdict |
|---|---|---|---|
| node-2 | NVRM 580.173.02 | `libnvidia-ml.so.580.178.04` | mismatch |
| node-4 | NVRM 580.173.02 | `libnvidia-ml.so.580.178.04` | mismatch |

`dpkg` on both carries `nvidia-driver-580 580.178.04-0ubuntu0.22.04.1`. The
package was upgraded under a running module. Node-2 additionally still carries
`nvidia-driver-535` / `libnvidia-compute-535`, left from the migration recorded
on 2026-08-15.

**Update 2026-09-14.** `compute-node-2` recovered on its own — its two T4s now
answer `nvidia-smi`. Usable capacity doubled from 2 to **4 × T4**. But
`node-3` (2 × A10 23 GB) and `node-4` (A40 46 GB) are still down, so **the
largest single card is 14,914 MiB** and the arbiter run's 30 GB requirement is
exactly as unmet as before. *More small cards do not unblock a large job.*

`W0.4`'s exit test was tightened accordingly: it now requires 3+ GPUs **and** at
least one with ≥ 20 GB free. Without that, recovering T4s would have flipped it
to PASS and made B1 look runnable when it is not — a plausible wrong status of
precisely the kind this programme exists to catch.

**Blocks:** Wave B entirely (arbiter + all seed replication), most of D, E, F —
approximately **760 of the register's 1,100 GPU-hours**. Seed replication (B4/B5/B6)
*is* now placeable on the four T4s; only the 30 GB arbiter is not.

**Fix, for the administrator:**

```bash
# On compute-node-2 and compute-node-4, as root:
systemctl isolate multi-user.target      # or simply reboot
rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
modprobe nvidia
nvidia-smi                               # must list the cards
# node-2 only — remove the superseded branch that confuses module selection:
apt-get purge -y 'nvidia-*-535' 'libnvidia-*-535'
apt-mark hold nvidia-driver-580
```

A reboot alone is very likely sufficient. Note both nodes were *already*
rebooted after the upgrade (uptimes 1 d 19 h and 1 d 14 h) and still mismatch,
which points at the 535 residue on node-2 and warrants checking
`/etc/modprobe.d/` and the initramfs on node-4.

**`sudo -n` fails for this account on both nodes.** This cannot be self-served.

**Escalation owner:** cluster administrator. **Raised:** W0.4, day 0.

---

## B-2 — HIGH: no Sri Lankan Tamil data, and SLCeleb not integrated

Every large Tamil corpus in use is **Indian** Tamil. The annual report §10.2
already forbids "any Sri Lankan Tamil claim". SLCeleb is the project's own
280-speaker si/ta corpus and remains the only one whose absolute EER would mean
what a reader assumes.

**Blocks:** the external validity of every Tamil number in the programme, and
the headline framing of all four papers.

**Mitigation now:** C6 (dialect gap) measures and *reports* the penalty rather
than pretending it is absent. That converts a blocker into a stated limitation,
which is the honest available move.

---

## B-3 — HIGH: four of nine proposals are undefined on the available conditions

Annual report Finding III-7. A1, A6, A7, A8 all need a domain variable; `si` and
`ta` each contain one language and one corpus. The methods **still train, still
converge, and still emit a plausible number** — A7 scored 0.663 % against a
0.623 % baseline while measuring nothing.

**Mitigation in force:** data-derived guards now `SystemExit` rather than run.
D2/D3 therefore target `combined` only. **Permanent fix is H2** — acquiring
within-speaker cross-channel Sri Lankan recordings unlocks all four at once.

---

## B-4 — MEDIUM: statistical power is capped by speaker count, not trial count

S = 91 held-out Sinhala, 131 Tamil. A single absolute EER carries ± 5–6 pp.
Differences of 0.09 pp are **not** resolvable; 2.0 pp differences are.

**Consequence for this plan:** every task in Wave B/D/E must report a **paired**
contrast on identical trials, and single-seed results stay flagged until Wave B4
lands. Rule 5 of the execution plan exists for this.

---

## B-5 — MEDIUM: the January–June 2026 record is not citable

Annual report §5 and §10.3 item 1. Those results are not reproducible from the
repository and are contradicted by the project's own July audit.

**Risk if unmanaged:** a number leaks into a paper through an intermediate
document that quoted it in good faith. **Mitigation:** G5 makes a single
authoritative statement and references it from every affected document, rather
than relying on each author remembering.

---

# 2. Risks, with the trigger that would confirm each

Carried forward from `MS §6` and extended with what this plan adds.

| # | Risk | Trigger that confirms it | Response |
|---|---|---|---|
| R1 | **91 Sinhala test speakers is not many.** Several proposals may produce effects inside the CI | B4 shows top-3 ordering unstable across seeds | Report intervals, not rankings; stop quoting single-seed wins |
| R2 | **The channel-confound hypothesis may be wrong** | D3 (A8) does not reverse PLDA's sign *and* A2 ≈ plain NAP | Move effort to label reliability, where A9 already shows signal |
| R3 | **The PEFT negative may not be safe to publish** | C5 finds the arms differ in two-stage-ness | Withdraw prediction 9's "Falsified" verdict pending a matched re-run |
| R4 | **A4/P3 depend on a corpus that must be built** | H4 has no progress by day 20 | Defer P3 explicitly rather than letting it slip silently |
| R5 | **Synthetic data is a trap here** — an SV model sits in the TTS reward loop | any TTS augmentation enters a training recipe | Evaluate on real held-out speakers only, and say so in the paper |
| R6 | **Deadlines move** | — | Confirm every venue against the official call before planning around it |
| R7 | **The cluster reboots mid-run** — four times in five days in August, and once more since | a run vanishes between two `--status` calls | Already mitigated: per-epoch checkpoints + auto-resume. **Verified today**: the five interrupted arms all retain resumable state |
| R8 | **A method undefined on the data emits a plausible number** | a guard is bypassed with `--force` | The guards `SystemExit`; overrides are written into the ledger |

---

# 3. The risk this plan itself introduces

**Nineteen idle days happened because a dispatcher died quietly.** This plan
answers with a one-task-at-a-time runner and a ledger — but that trades one
failure mode for another: a plan that requires an operator at every step stalls
the moment the operator is busy.

**The honest mitigation** is that the runner is *resumable and inspectable*, not
that it is attended. `runner.py --status` reconstructs the truth from the
filesystem the way each strand's `scaffold.py --check` does, so it cannot drift
from what actually exists. If nobody looks for a week, the answer to "what
happened" is still one command — which is precisely what was missing on
24 August.
