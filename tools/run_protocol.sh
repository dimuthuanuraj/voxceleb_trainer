#!/usr/bin/env bash
# =====================================================================
# SL-SPV master experimental protocol — every training command in
# execution order. Lines are individually copy-pasteable; the script
# can also be run end-to-end (~5-7 days wall-clock across A40/2xT4/A10).
#
# Pre-flight requirements:
#   1. paths.env exists with SL_SPV_DATA_ROOT set; `source paths.env`
#   2. Corpus loaded under $SL_SPV_DATA_ROOT/sl_celeb/<lang>/<spk>/<utt>.wav
#   3. MUSAN + RIRS_NOISES present
#   4. English VoxCeleb1 ECAPA checkpoint at
#      $SL_SPV_DATA_ROOT/checkpoints/ecapa_voxceleb1.model
#      (required for P1, P2, fine-tune-only, and 4 of the 5 ablations)
#   5. python tools/sl_dataprep.py --corpus_root ... --out_dir ...
#      has been run (generates lists/, lookup files, cohort, plda list)
# =====================================================================

set -euo pipefail

# ---- Pre-flight checks ----------------------------------------------
if [[ -z "${SL_SPV_DATA_ROOT:-}" ]]; then
    echo "ERROR: SL_SPV_DATA_ROOT not set. Run: source paths.env"
    exit 1
fi

LISTS_DIR="$SL_SPV_DATA_ROOT/sl_celeb/lists"
SPEAKERS_CSV="$LISTS_DIR/speakers.csv"
if [[ ! -f "$SPEAKERS_CSV" ]]; then
    echo "ERROR: $SPEAKERS_CSV not found. Run tools/sl_dataprep.py first."
    exit 1
fi

NCLASSES=$(awk -F, 'END{print NR-1}' "$SPEAKERS_CSV")
echo "[run_protocol] NCLASSES=$NCLASSES (from $SPEAKERS_CSV)"

PRIMARY_SEEDS=(42 123 7)
SEED_42=(42)

# =====================================================================
# Step 1 — PRIMARY CONFIGURATIONS (3 configs × 3 seeds = 9 runs)
# Headline numbers. Reported as mean ± std in the thesis/paper.
# =====================================================================

echo ""
echo "==============================="
echo "[1/3] PRIMARY CONFIGURATIONS"
echo "==============================="

for SEED in "${PRIMARY_SEEDS[@]}"; do
    # ---------- P0 ECAPA from scratch ----------
    echo "[run] P0 baseline seed=$SEED"
    python trainSpeakerNet.py \
        --config configs/sl_p0_baseline.yaml \
        --seed "$SEED" \
        --nClasses "$NCLASSES" \
        --save_path "exps/P0_baseline_seed${SEED}"

    # ---------- P1 cross-lingual fine-tune ----------
    echo "[run] P1 fine-tune seed=$SEED"
    python trainSpeakerNet.py \
        --config configs/sl_p1_finetune.yaml \
        --seed "$SEED" \
        --nClasses "$NCLASSES" \
        --save_path "exps/P1_finetune_seed${SEED}"

    # ---------- P2 full feature stack ----------
    echo "[run] P2 full-stack seed=$SEED"
    python trainSpeakerNet.py \
        --config configs/sl_full_stack.yaml \
        --seed "$SEED" \
        --nClasses "$NCLASSES" \
        --save_path "exps/P2_full_stack_seed${SEED}"
done

# =====================================================================
# Step 2 — FEATURE-ADD CONFIGURATIONS (4 configs × 1 seed = 4 runs)
# Each adds ONE feature to P0. Measures STANDALONE contribution.
# =====================================================================

echo ""
echo "==============================="
echo "[2/3] FEATURE-ADD CONFIGURATIONS"
echo "==============================="

FEATURE_CONFIGS=(
    sl_feature_finetune_only
    sl_feature_lang_aux_only
    sl_feature_as_norm_only
    sl_feature_plda_only
)
for CFG in "${FEATURE_CONFIGS[@]}"; do
    echo "[run] feature-add: $CFG"
    python trainSpeakerNet.py \
        --config "configs/${CFG}.yaml" \
        --seed 42 \
        --nClasses "$NCLASSES" \
        --save_path "exps/${CFG}_seed42"
done

# =====================================================================
# Step 3 — P2 ABLATION CONFIGURATIONS (5 configs × 1 seed = 5 runs)
# Each removes ONE feature from P2. Measures MARGINAL contribution.
# =====================================================================

echo ""
echo "==============================="
echo "[3/3] P2 ABLATION CONFIGURATIONS"
echo "==============================="

ABLATION_CONFIGS=(
    sl_p2_no_finetune
    sl_p2_no_llrd
    sl_p2_no_lang_aux
    sl_p2_no_as_norm
    sl_p2_no_plda
)
for CFG in "${ABLATION_CONFIGS[@]}"; do
    echo "[run] ablation: $CFG"
    python trainSpeakerNet.py \
        --config "configs/${CFG}.yaml" \
        --seed 42 \
        --nClasses "$NCLASSES" \
        --save_path "exps/${CFG}_seed42"
done

# =====================================================================
# Step 4 — AGGREGATE RESULTS
# Markdown table to stdout; LaTeX table to thesis_results_table.tex
# =====================================================================

echo ""
echo "==============================="
echo "[4/4] AGGREGATE & EXPORT"
echo "==============================="

# Primary multi-seed aggregation (mean ± std)
echo ""
echo "--- Primary configs (mean ± std over 3 seeds) ---"
python tools/aggregate_seeds.py \
    --prefixes P0_baseline P1_finetune P2_full_stack

# LaTeX output for direct thesis paste
python tools/aggregate_seeds.py --latex \
    --prefixes P0_baseline P1_finetune P2_full_stack \
    > thesis_results_table.tex
echo "Wrote thesis_results_table.tex (paste into thesis_chapters/04_results.tex)"

# Single-seed aggregation for feature-add + ablation
echo ""
echo "--- Feature-add configs (seed 42 only) ---"
python tools/aggregate_seeds.py \
    --prefixes sl_feature_finetune_only sl_feature_lang_aux_only \
               sl_feature_as_norm_only sl_feature_plda_only

echo ""
echo "--- P2 ablation configs (seed 42 only) ---"
python tools/aggregate_seeds.py \
    --prefixes sl_p2_no_finetune sl_p2_no_llrd sl_p2_no_lang_aux \
               sl_p2_no_as_norm sl_p2_no_plda

echo ""
echo "==============================="
echo "[DONE] All 18 runs complete."
echo "==============================="
