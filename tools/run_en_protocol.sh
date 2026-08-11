#!/usr/bin/env bash
# =====================================================================
# English VoxCeleb2 training protocol — exercises every UNIVERSAL
# feature (ECAPA, LLRD, AS-Norm, PLDA, SSL frontend, deterministic /
# mixed-precision plumbing) end-to-end on real data, and also produces
# the English ECAPA checkpoint that the SL fine-tune configs consume.
#
# Three configs, in order:
#   1. en_p0_baseline.yaml   ECAPA from scratch (80 ep) — baseline EER
#                            + produces the checkpoint sl_p1/sl_full
#                            need at $SL_SPV_DATA_ROOT/checkpoints/
#                            ecapa_voxceleb1.model
#   2. en_full.yaml          Continues from #1; +LLRD +AS-Norm +PLDA
#                            (30 epochs) — measures universal-feature
#                            uplift over baseline.
#   3. en_ssl_wavlm.yaml     Frozen WavLM-Base frontend, head only
#                            (60 epochs) — exercises FEATURE-001 on
#                            the corpus WavLM saw at pretrain time.
#
# Pre-flight:
#   1. `conda activate SL_SPV` is active
#   2. Run from the voxceleb_trainer/ directory
#   3. voxceleb_trainer/data/ contains the three symlinks (see SETUP.md §7):
#        data/voxceleb_new   → /path/to/voxceleb_new
#        data/musan          → /path/to/musan
#        data/RIRS_NOISES    → /path/to/RIRS_NOISES
#   4. data/voxceleb_new contains:
#        train_list.txt  test_list.txt  voxceleb2/  voxceleb1/
#   5. python tools/en_dataprep.py has been run (writes the AS-Norm
#      cohort + PLDA training list into data/voxceleb_new/lists/)
# =====================================================================

set -euo pipefail

# ---- Pre-flight checks ----------------------------------------------
for f in \
    "data/voxceleb_new/train_list.txt" \
    "data/voxceleb_new/test_list.txt"  \
    "data/voxceleb_new/voxceleb2"      \
    "data/voxceleb_new/voxceleb1"      \
    "data/musan"                       \
    "data/RIRS_NOISES/simulated_rirs"
do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: missing required path: $f"
        echo "       Check that data/ symlinks exist (see SETUP.md §7)."
        exit 1
    fi
done

LISTS_DIR="data/voxceleb_new/lists"
if [[ ! -f "$LISTS_DIR/asnorm_cohort.txt" || ! -f "$LISTS_DIR/plda_train_list.txt" ]]; then
    echo "ERROR: en_dataprep outputs missing under $LISTS_DIR"
    echo "Run:"
    echo "  python tools/en_dataprep.py \\"
    echo "      --vox_root    data/voxceleb_new \\"
    echo "      --train_list  data/voxceleb_new/train_list.txt \\"
    echo "      --out_dir     data/voxceleb_new/lists \\"
    echo "      --cohort_size 5000 --plda_speakers 1000 --seed 42"
    exit 1
fi

# ---- Stage 1: ECAPA baseline (from scratch, 80 epochs) --------------
echo "===================================================================="
echo "[1/3] en_p0_baseline.yaml — ECAPA from scratch on VoxCeleb2"
echo "===================================================================="
python trainSpeakerNet.py \
    --config configs/en_p0_baseline.yaml \
    --nClasses 5994

# Promote the best checkpoint to the path SL fine-tune configs expect.
# (SL configs still reference ${SL_SPV_DATA_ROOT}/checkpoints/... — adjust
#  this copy step to wherever your SL checkpoints directory lives.)
if [[ -f "exps/EN_p0_baseline_seed42/model/best.model" ]]; then
    if [[ -n "${SL_SPV_DATA_ROOT:-}" ]]; then
        mkdir -p "$SL_SPV_DATA_ROOT/checkpoints"
        cp -v "exps/EN_p0_baseline_seed42/model/best.model" \
              "$SL_SPV_DATA_ROOT/checkpoints/ecapa_voxceleb1.model"
    else
        echo "[info] SL_SPV_DATA_ROOT not set — skipping checkpoint copy."
        echo "       Manually copy exps/EN_p0_baseline_seed42/model/best.model"
        echo "       to wherever your SL configs expect ecapa_voxceleb1.model."
    fi
fi

# ---- Stage 2: Full universal features (continuation, 30 epochs) -----
echo "===================================================================="
echo "[2/3] en_full.yaml — LLRD + AS-Norm + PLDA continuation"
echo "===================================================================="
python trainSpeakerNet.py \
    --config configs/en_full.yaml \
    --nClasses 5994

# ---- Stage 3: SSL frontend (WavLM-Base, 60 epochs) ------------------
echo "===================================================================="
echo "[3/3] en_ssl_wavlm.yaml — frozen WavLM-Base frontend"
echo "===================================================================="
python trainSpeakerNet.py \
    --config configs/en_ssl_wavlm.yaml \
    --nClasses 5994

echo "===================================================================="
echo "EN protocol complete."
echo "  baseline:  exps/EN_p0_baseline_seed42/"
echo "  full:      exps/EN_full_seed42/"
echo "  ssl wavlm: exps/EN_ssl_wavlm_seed42/"
echo "===================================================================="
