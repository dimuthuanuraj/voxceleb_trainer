#!/usr/bin/env bash
# C1 -- The short-duration table: 2 s / 3 s / 5 s test utterances.
#
# `MASTER_SYNTHESIS.md` section 5: "Truncate test utterances to 2 s / 3 s / 5 s
# across existing Stage A checkpoints (category 03, ERes2NetV2). No training.
# Nobody has published it for Sinhala or Tamil."
#
# It is an operating-point table: a deployed verifier is handed whatever the
# speaker said, and the 3 s evaluation segment the benchmark uses is a choice,
# not a law. Every paper in the roadmap gets a stronger deployment section from
# this, and it costs no training.
#
# NOTE the interaction with T5's cost finding: at 2 s a crop is 50 tokens, at 8 s
# it is 200. So this table also samples the regime where the O(T^2) term starts
# to matter -- 128 tokens is ~5.1 s.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/C1"; mkdir -p "$OUTDIR"
DURATIONS=(200 300 500)      # --eval_frames, in 10 ms frames: 2 s / 3 s / 5 s
SYSTEMS=(
  F_ssl_mhubert_lw_aamsoftmax_si_s42
  F_ssl_mhubert_lw_aamsoftmax_ta_s42
  A_ecapa512_aamsoftmax_si_s42
  A_ecapa512_aamsoftmax_ta_s42
)

require_gpu_mb 8000
trainer_guard_snapshot
cd "$EXPERIMENTS"

for sys_id in "${SYSTEMS[@]}"; do
    [[ -d "results/$sys_id" ]] || { warn "missing experiment $sys_id -- skipping"; continue; }
    for fr in "${DURATIONS[@]}"; do
        step "$sys_id @ ${fr} frames ($((fr/100)) s)"
        set +e
        bash "$GPURUN" -m 8000 -- python tools/evaluate.py \
            --exp "$sys_id" --device cuda --eval-frames "$fr" \
            2>&1 | tee "$OUTDIR/${sys_id}_${fr}.log"
        log "  rc=${PIPESTATUS[0]}"
        set -e
    done
done

trainer_guard_verify

step "Table"
python3 - "$OUTDIR" "${SYSTEMS[@]}" <<'PY'
import glob, json, os, re, sys
outdir, *systems = sys.argv[1:]
table = {}
for log in sorted(glob.glob(os.path.join(outdir, "*_*.log"))):
    base = os.path.basename(log)[:-4]
    m = re.match(r"(.+)_(\d+)$", base)
    if not m:
        continue
    sys_id, fr = m.group(1), int(m.group(2))
    txt = open(log, errors="replace").read()
    eers = re.findall(r"EER[^\d\-]{0,12}([\d.]+)", txt)
    if eers:
        table.setdefault(sys_id, {})[fr] = float(eers[-1])
print(f"  {'system':48s} " + "".join(f"{d//100:>7d}s" for d in (200, 300, 500)))
for s in systems:
    if s in table:
        row = "".join(f"{table[s].get(d, float('nan')):8.3f}" for d in (200, 300, 500))
        print(f"  {s:48s} {row}")
ok = bool(table)
json.dump({"task": "C1", "ok": ok, "eer_by_eval_frames": table,
           "note": "eval_frames are 10 ms frames: 200=2s, 300=3s (benchmark default), 500=5s"},
          open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
if not ok:
    print("  No EER parsed. Check whether tools/evaluate.py accepts --eval-frames;")
    print("  if not, pass it through the experiment's argv instead.")
PY
