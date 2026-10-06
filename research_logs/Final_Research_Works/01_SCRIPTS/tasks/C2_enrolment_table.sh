#!/usr/bin/env bash
# C2 -- The enrolment / privacy table: N = 1, 5, 10, 15 enrolment utterances.
#
# MASTER_SYNTHESIS section 5: "Report deployment numbers for N = 1, 5, 10, 15
# enrolment utterances. Costs no training, uses embeddings already in the
# database, strengthens the accuracy claim, and produces the analysis a KYC
# deployment will eventually be asked for."
#
# Why it is called the privacy table: the fewer utterances a system needs to
# enrol a person, the less voice data has to be retained. N is a privacy
# parameter as much as an accuracy one, and a deployment is entitled to know the
# exchange rate. SL_SPV/voiceid is a live biometric product, so this is not
# hypothetical.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/C2"; mkdir -p "$OUTDIR"
require_gpu_mb 8000
trainer_guard_snapshot
cd "$EXPERIMENTS"

SYSTEMS=(F_ssl_mhubert_lw_aamsoftmax_si_s42 F_ssl_mhubert_lw_aamsoftmax_ta_s42)

step "Extracting embeddings once per system"
for s in "${SYSTEMS[@]}"; do
    [[ -d "results/$s" ]] || { warn "missing $s"; continue; }
    set +e
    bash "$GPURUN" -m 8000 -- python tools/extract_embeddings.py --exp "$s" --device cuda \
        2>&1 | tee "$OUTDIR/${s}_embed.log"
    log "  rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

step "Multi-utterance enrolment sweep"
python3 - "$OUTDIR" "${SYSTEMS[@]}" <<'PY'
"""Average N enrolment embeddings per speaker and re-score.

Averaging in embedding space before length-norm is the standard multi-enrolment
back end and is what the deployed product does. The trial list is unchanged, so
the only variable is how many utterances the enrolment side pools.
"""
import glob, itertools, json, os, sys
import numpy as np

outdir, *systems = sys.argv[1:]
EXPD = os.environ["EXPERIMENTS"]
rng = np.random.default_rng(42)
NS = [1, 5, 10, 15]
table, note = {}, None

def eer_from(scores, labels):
    order = np.argsort(-scores)
    lab = labels[order]
    tgt, non = lab.sum(), (1 - lab).sum()
    if not tgt or not non:
        return None
    fn = tgt - np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    fnr, fpr = fn / tgt, fp / non
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2 * 100)

for s in systems:
    cands = glob.glob(os.path.join(EXPD, "results", s, "*embed*.npz")) + \
            glob.glob(os.path.join(EXPD, "results", s, "embeddings", "*.npz"))
    if not cands:
        note = "no embedding archive found; run tools/extract_embeddings.py first"
        print(f"  {s}: {note}")
        continue
    z = np.load(cands[0], allow_pickle=True)
    keys = list(z.keys())
    if not {"embeddings", "speakers"} <= set(keys):
        print(f"  {s}: archive keys are {keys}; expected embeddings+speakers")
        continue
    emb = np.asarray(z["embeddings"], dtype=np.float64)
    spk = np.asarray(z["speakers"])
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    uniq = np.unique(spk)
    print(f"\n  {s}: {len(emb)} utts, {len(uniq)} speakers")
    row = {}
    for N in NS:
        cents, keep = [], []
        for sp in uniq:
            idx = np.flatnonzero(spk == sp)
            if len(idx) < N + 1:            # need N to enrol + >=1 to test
                continue
            pick = rng.choice(idx, N, replace=False)
            c = emb[pick].mean(0)
            cents.append(c / (np.linalg.norm(c) + 1e-9))
            keep.append((sp, np.setdiff1d(idx, pick)))
        if len(cents) < 2:
            continue
        C = np.stack(cents)
        sc, lb = [], []
        for i, (sp, rest) in enumerate(keep):
            for j in rest[:3]:
                sims = C @ emb[j]
                sc.extend(sims.tolist())
                lb.extend((np.arange(len(C)) == i).astype(int).tolist())
        e = eer_from(np.asarray(sc), np.asarray(lb))
        row[N] = e
        print(f"    N={N:2d}  speakers={len(C):3d}  EER={e:.3f} %" if e is not None
              else f"    N={N:2d}  insufficient")
    table[s] = row

print("\n  " + "=" * 56)
print(f"  {'system':44s}" + "".join(f"{n:>7d}" for n in NS))
for s, row in table.items():
    print(f"  {s[:44]:44s}" + "".join(
        f"{row.get(n, float('nan')):7.3f}" for n in NS))

json.dump({"task": "C2", "ok": bool(table), "eer_by_n_enrolment": table,
           "n_values": NS, "note": note,
           "method": "L2-normalised mean of N enrolment embeddings, cosine scoring; "
                     "held-out utterances only, speakers with < N+1 utterances dropped"},
          open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY

cat <<'NOTE'

  READ WITH THE POWER CEILING IN MIND. Dropping speakers with fewer than N+1
  utterances shrinks S as N grows, so the N=15 column rests on fewer speakers
  than the N=1 column and its interval is WIDER even though its EER is lower.
  Report the speaker count per column alongside the EER, or the table will read
  as a cleaner monotone trend than the evidence supports.
NOTE
