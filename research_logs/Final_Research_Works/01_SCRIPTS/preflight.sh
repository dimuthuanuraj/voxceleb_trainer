#!/usr/bin/env bash
# W0.1 — Preflight. Run this first, and again after every cluster reboot.
#
# Checks, in order of what actually breaks here:
#   1. cluster GPUs that really answer nvidia-smi (not just ones that ping)
#   2. the conda env exists on every live node (envs are per-node)
#   3. experiments/splits/ integrity -- the open-set lists, NOT the shipped
#      lists/, which are 100 % closed-set (annual report 6.8.1)
#   4. disk headroom on the NFS shares
#   5. the trainer tree is clean (isolation rule 6)
#
# Writes 02_STATE/preflight.json. Exit test: .ok == true

. "$(dirname "$0")/lib/common.sh"

OUT="$STATE/preflight.json"
OK=true
declare -a NOTES=()

step "1. Cluster GPUs"
GPU_LINES="$(probe_gpus || true)"
if [[ -z "$GPU_LINES" ]]; then
    warn "no usable GPU anywhere on the cluster"
    OK=false; NOTES+=("no usable GPU -- blocker B-1")
else
    printf '%s\n' "$GPU_LINES" | while read -r node gpu free; do
        log "  $node gpu$gpu  ${free} MiB free"
    done
fi
NGPU=$(printf '%s\n' "$GPU_LINES" | grep -c . || true)
MAXFREE=$(printf '%s\n' "$GPU_LINES" | awk '{if($3>m)m=$3}END{print m+0}')
log "usable GPUs: $NGPU   largest free: ${MAXFREE} MiB"
if (( MAXFREE < 30000 )); then
    warn "no GPU can hold the arbiter run (needs 30000 MiB) -- Wave B is blocked"
    NOTES+=("arbiter blocked: largest GPU ${MAXFREE} MiB < 30000")
fi

step "2. Dead nodes, with the reason"
for n in compute-node-1 compute-node-2 compute-node-3 compute-node-4; do
    if ! timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=6 "$n" true 2>/dev/null; then
        warn "  $n UNREACHABLE"; NOTES+=("$n unreachable"); continue
    fi
    if ! timeout 15 ssh -o BatchMode=yes "$n" 'nvidia-smi -L' >/dev/null 2>&1; then
        krn=$(timeout 10 ssh -o BatchMode=yes "$n" \
              "awk '{print \$8}' /proc/driver/nvidia/version 2>/dev/null" 2>/dev/null || echo "?")
        usr=$(timeout 10 ssh -o BatchMode=yes "$n" \
              "ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.* 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1" 2>/dev/null || echo "?")
        warn "  $n driver mismatch: kernel module $krn vs userspace $usr"
        NOTES+=("$n driver mismatch $krn vs $usr -- needs root reload/reboot (W0.4)")
    fi
done

step "3. Conda env on live nodes"
printf '%s\n' "$GPU_LINES" | awk '{print $1}' | sort -u | while read -r n; do
    [[ -n "$n" ]] || continue
    if timeout 15 ssh -o BatchMode=yes "$n" \
        "[ -d \$HOME/anaconda2025/envs/$CONDA_ENV ]" 2>/dev/null; then
        log "  $n: env $CONDA_ENV present"
    else
        warn "  $n: env $CONDA_ENV MISSING (envs are per-node)"
    fi
done

step "4. Open-set splits integrity"
SPLITS="$EXPERIMENTS/splits"
if [[ ! -d "$SPLITS" ]]; then
    warn "splits directory missing: $SPLITS"; OK=false; NOTES+=("splits missing")
else
    NSPLIT=$(find "$SPLITS" -maxdepth 1 -mindepth 1 -type d | wc -l)
    log "  $NSPLIT split directories under experiments/splits/"
    for s in si_pooled combined_si_ta slr52_sinhala slr127_tamil nisp_tamil en_matched; do
        if [[ -d "$SPLITS/$s" ]]; then log "    $s OK"; else warn "    $s MISSING"; fi
    done
    if [[ -f "$SPLITS/splits_report.json" ]]; then
        python3 - "$SPLITS/splits_report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
n = len(d) if isinstance(d, list) else len(d.get("splits", d))
print(f"    splits_report.json: {n} entries, SHA-256 recorded per emitted list")
PY
    fi
    warn "  REMINDER: use experiments/splits/ only. The shipped lists/ are 100% closed-set."
fi

step "5. Disk headroom"
df -h /mnt/ricproject3 /mnt/ricproject5 2>/dev/null | sed 's/^/  /' || warn "df failed"
AVAIL_K=$(df -k /mnt/ricproject3 2>/dev/null | awk 'NR==2{print $4}')
if [[ -n "$AVAIL_K" ]] && (( AVAIL_K < 50*1024*1024 )); then
    warn "  less than 50 GiB free on /mnt/ricproject3"
    NOTES+=("low disk on ricproject3")
fi

step "6. Trainer isolation"
DIRTY=$( ( cd "$TRAINER" && git status --porcelain 2>/dev/null ) | grep -vE '^\?\?' | wc -l )
log "  voxceleb_trainer/ tracked modifications: $DIRTY"
(( DIRTY > 0 )) && NOTES+=("voxceleb_trainer has $DIRTY tracked modifications -- review before running")

step "7. Orphaned trained checkpoints (work already paid for)"
python3 - <<'PY'
import glob, os, re, json
SL = os.environ.get("SL_SPV", "/mnt/ricproject3/2026/SL_SPV")
rows = []
for pat in ("proposals/A*/out/exps/*/model", "transformer_sv/T*/out/exps/*/model"):
    for m in sorted(glob.glob(os.path.join(SL, pat))):
        tag = os.path.basename(os.path.dirname(m))
        ck = glob.glob(os.path.join(m, "model0*.model"))
        if not ck:
            continue
        # m = <strand>/out/exps/<tag>/model  ->  out_dir must be <strand>/out
        out_dir = os.path.dirname(os.path.dirname(os.path.dirname(m)))
        scored = bool(glob.glob(os.path.join(out_dir, f"{tag}*.eval.json"))) or \
                 bool(glob.glob(os.path.join(out_dir, "scores", tag, "*")))
        eers = {}
        for f in glob.glob(os.path.join(m, "model0*.eer")):
            try:
                eers[int(re.search(r"model0*(\d+)\.eer", f).group(1))] = float(open(f).read().split()[0])
            except Exception:
                pass
        best = min(eers, key=eers.get) if eers else None
        rows.append((tag, len(ck), max(eers) if eers else 0, best,
                     eers[best] if best else None, scored))
print(f"  {'tag':42s} {'ep':>3s} {'best@':>5s} {'val':>7s}  scored")
for tag, n, last, best, v, sc in rows:
    flag = "yes" if sc else "NO  <-- unscored"
    print(f"  {tag:42s} {last:3d} {best or 0:5d} {v if v is not None else 0:7.4f}  {flag}")
unscored = [r for r in rows if not r[5]]
print(f"\n  {len(unscored)} trained/partial run(s) with checkpoints and NO score.")
PY

step "Summary"
python3 - "$OUT" "$OK" "$NGPU" "$MAXFREE" "${NOTES[@]:-}" <<'PY'
import json, sys, datetime
out, ok, ngpu, maxfree, *notes = sys.argv[1:]
notes = [n for n in notes if n.strip()]
d = {"ok": ok == "true",
     "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
     "usable_gpus": int(ngpu or 0),
     "largest_free_mb": int(maxfree or 0),
     "arbiter_placeable": int(maxfree or 0) >= 30000,
     "notes": notes}
json.dump(d, open(out, "w"), indent=2)
print(json.dumps(d, indent=2))
PY

log "preflight written to $OUT"
