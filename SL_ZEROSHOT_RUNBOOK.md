# SL Zero-Shot Benchmark Runbook (P0 + P1)

**Created:** 2026-07-03 · **Validated on:** head-node + compute-node-4 (A40)
Reproduces the Sinhala/Tamil benchmark build and the zero-shot baseline table
from scratch. Companion docs:
- `research_logs/2026-07-03-project-audit-sota-roadmap.md` — why these steps (audit + roadmap P0–P6)
- `research_logs/2026-07-03-sl-benchmark-v0-design.md` — protocol design decisions
- `research_logs/2026-07-03-zeroshot-baseline-table.md` — results log

---

## 0. Environment

Everything runs in the existing conda env `SL_SPV` (Python 3.11, torch 2.5.1):

```bash
PY=/home/anuraj/anaconda2025/envs/SL_SPV/bin/python
# One-time additions made on 2026-07-03:
$PY -m pip install speechbrain gdown
```

Key paths:

| What | Where |
|---|---|
| Repo | `/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer` |
| Raw corpora | `/mnt/ricproject3/2025/data/sl_corpora/{slr52,slr65,slceleb}` |
| Benchmark corpus (symlinks) | `/mnt/ricproject3/2025/data/sl_celeb/<lang>/<spk>/<utt>` |
| Trial lists | `/mnt/ricproject3/2025/data/sl_celeb/lists/` |
| Results + embedding cache | `/mnt/ricproject3/2025/data/sl_celeb/{results,emb_cache}/` |
| GPU staging (shared /home NFS) | `/home/anuraj/sl_spv_bench/` |

## 1. Download corpora (head node, ~16 GB, ~run once)

```bash
mkdir -p /mnt/ricproject3/2025/data/sl_corpora/{slr52,slr65}

# SLR52 — Sinhala, 478 usable speakers, 185k utts (16 zips + speaker TSV)
cd /mnt/ricproject3/2025/data/sl_corpora/slr52
wget https://www.openslr.org/resources/52/utt_spk_text.tsv
for h in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  wget -c "https://www.openslr.org/resources/52/asr_sinhala_$h.zip"
done
for h in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  unzip -o -q asr_sinhala_$h.zip -d extracted     # -> extracted/asr_sinhala/...
done

# SLR65 — Tamil, 49 usable speakers, 4.3k utts
cd ../slr65
wget https://www.openslr.org/resources/65/{line_index_female.tsv,line_index_male.tsv,LICENSE}
wget -c https://www.openslr.org/resources/65/{ta_in_female.zip,ta_in_male.zip}
unzip -o -q ta_in_female.zip -d extracted
unzip -o -q ta_in_male.zip  -d extracted          # -o: both zips carry line_index.tsv
```

Gotchas encountered:
- SLR52's speaker TSV lives at `extracted/asr_sinhala/utt_spk_text.tsv` — pass
  the **nested** dir to the ingest tool, not `extracted/`.
- SLR65 male files are prefixed `tag_` (not `tam_`); female `taf_`. The ingest
  tool knows this.

## 2. Build the corpus layout (symlinks, no copies)

```bash
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer
python3 tools/ingest_openslr.py \
  --slr52_dir /mnt/ricproject3/2025/data/sl_corpora/slr52/extracted/asr_sinhala \
  --slr65_dir /mnt/ricproject3/2025/data/sl_corpora/slr65/extracted \
  --out_root  /mnt/ricproject3/2025/data/sl_celeb \
  --min_utts 10
# Expected: si 478 speakers / ~185k utts, ta 49 / 4,286.
# Writes ingest_manifest.csv + spk_meta.csv (gender known for ta only).
```

## 3. Generate trial lists (seed-reproducible)

```bash
python3 tools/sl_dataprep.py \
  --corpus_root /mnt/ricproject3/2025/data/sl_celeb \
  --out_dir     /mnt/ricproject3/2025/data/sl_celeb/lists \
  --langs si ta --test_frac 0.2 \
  --target_pairs_per_lang 3000 --impostor_ratio 3 \
  --cohort_size 500 \
  --spk_meta /mnt/ricproject3/2025/data/sl_celeb/spk_meta.csv \
  --seed 42
# Expected: 12,000 pairs per language; test_list_cs.txt = 0 pairs (OpenSLR has
# no bilingual speakers — SLCeleb will fill this); asnorm_cohort.txt = 500;
# train_list.txt ~151k utts, nClasses 527.
```

Protocol flags added 2026-07-03 (see benchmark-design doc):
`--session_from parentdir|regex` (cross-session targets — use for SLCeleb;
OpenSLR is flat/single-session so v0 uses `none`), `--spk_meta` (same-gender
impostors), automatic pair dedup.

## 4. Run the zero-shot evaluation

### 4a. CPU (head node) — fine for smoke tests (~4 files/s)

```bash
$PY tools/zeroshot_eval.py \
  --corpus_root /mnt/ricproject3/2025/data/sl_celeb \
  --trials /mnt/ricproject3/2025/data/sl_celeb/lists/test_list_ta.txt \
  --models speechbrain_ecapa redimnet:b1 \
  --cache_dir /mnt/ricproject3/2025/data/sl_celeb/emb_cache \
  --out /mnt/ricproject3/2025/data/sl_celeb/results/smoke.json --device cpu
```

### 4b. GPU (compute nodes) — full runs (~70 files/s on the A40)

The compute nodes do **not** currently mount `/mnt/ricproject*` (fstab entries
exist; remount needs sudo: `sudo mount /mnt/ricproject3` — this unmount is
also what killed the May EN-baseline runs). `/home` **is** NFS-shared, so
stage the test audio there:

```bash
# Head node: collect unique test files and stage (~1.8 GB for v0 lists)
cd /mnt/ricproject3/2025/data/sl_celeb
cat lists/test_list_si.txt lists/test_list_ta.txt \
  | awk '{print $2"\n"$3}' | sort -u > /tmp/test_files.txt
mkdir -p /home/anuraj/sl_spv_bench/{audio,code,results,emb_cache}
rsync -aL --files-from=/tmp/test_files.txt . /home/anuraj/sl_spv_bench/audio/  # -L resolves symlinks
cp lists/test_list_{si,ta}.txt /home/anuraj/sl_spv_bench/
cp /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/tools/zeroshot_eval.py \
   /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/tuneThreshold.py \
   /home/anuraj/sl_spv_bench/code/

# Launch on a GPU node (node4 = A40 48GB; node3 = 2xA10; nodes 1-2 = 2xT4)
ssh compute-node-4 'cd /home/anuraj/sl_spv_bench && \
  nohup /home/anuraj/anaconda2025/envs/SL_SPV/bin/python code/zeroshot_eval.py \
    --corpus_root audio \
    --trials test_list_si.txt test_list_ta.txt \
    --models speechbrain_ecapa redimnet:b1 redimnet:b2 redimnet:b6 \
    --cache_dir emb_cache --out results/zeroshot_v0_full.json \
    --device cuda > results/run.log 2>&1 &'
tail -f /home/anuraj/sl_spv_bench/results/run.log   # ~5 min/model for 18k files

# Copy results back
cp /home/anuraj/sl_spv_bench/results/zeroshot_v0_full.{json,md} \
   /mnt/ricproject3/2025/data/sl_celeb/results/
```

Notes:
- Model checkpoints cache to `~/.cache/{torch/hub,huggingface}` — shared via
  /home, so a model downloaded once (even on CPU) is reusable on every node.
- Embeddings cache per model in `emb_cache/<model>.npz`; adding a new trial
  list re-uses them (only new files are extracted).
- If SSH warns "REMOTE HOST IDENTIFICATION HAS CHANGED": nodes were reimaged;
  clear the stale keys with `ssh-keygen -R compute-node-N`.

## 5. Reference results (v0, 2026-07-03, 12k trials/lang, cosine, no norm/calib)

| Model | Params | Vox1-O EER | si EER_avg | ta EER_avg | si minDCF .01/.05 | ta minDCF .01/.05 |
|---|---|---|---|---|---|---|
| redimnet:b6 | 15M | 0.37% | **2.71%** | **1.48%** | 0.238 / 0.153 | 0.422 / 0.160 |
| redimnet:b1 | 2.2M | 0.73% | 3.57% | 1.67% | 0.306 / 0.209 | 0.406 / 0.162 |
| redimnet:b2 | 4.7M | 0.52% | 4.17% | 1.97% | 0.422 / 0.244 | 0.454 / 0.170 |
| speechbrain_ecapa | ~20M | 0.80% | 4.78% | 3.21% | 0.471 / 0.291 | 0.482 / 0.207 |

Raw: `/mnt/ricproject3/2025/data/sl_celeb/results/zeroshot_v0_full.{json,md}`.
Read with the v0 caveats (single-session read speech → optimistic; relative
comparison only — see benchmark-design doc §5).

## 6. Extending the run

- **Add a model:** new spec branch in `EmbeddingBackend._build()` in
  `tools/zeroshot_eval.py`; everything else (caching, scoring, tables) is
  automatic. Planned: `wespeaker:*` (untested stub exists), WavLM+ECAPA via
  ESPnet-SPK.
- **Add SLCeleb (v1):** public Drive folder
  `https://drive.google.com/drive/folders/1A_INdaAl-16mMscOpzO-Qcj37rOgfOKE`
  (structure `SLCeleb/<lang>/<dev|test>/<spk_id>/<genre>/<genre>-<video>-<utt>.wav`).
  gdown gets rate-limited on per-file fetches — prefer the group's original
  archives, or resume gdown in chunks. Then: add a `collect_slceleb()` to
  `tools/ingest_openslr.py`, re-run steps 2–4 with
  `--session_from regex --session_regex '([a-z]+-[0-9]+)-[0-9]+\.wav$'`
  so targets are cross-video, and the cs (cross-lingual) list becomes non-empty
  for bilingual speakers.
- **Re-generate lists with a different seed/volume:** step 3 only; embedding
  cache remains valid (same audio files).
