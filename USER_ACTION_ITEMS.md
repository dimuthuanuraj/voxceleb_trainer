# Immediate Action Items (for Dimuthu) — updated 2026-07-03

Things only you can do; everything else is automated or already running.
Tick these off and the next phases unblock.

## 1. Place the SLCeleb audio (unblocks benchmark v1 — the publication-grade eval)

Copy the finalized SLCeleb dataset (280 speakers, ~34k wavs) to:

```
/mnt/ricproject3/2025/data/sl_corpora/slceleb/SLCeleb/<sinhala|tamil>/<dev|test>/<spk_id>/<genre>/*.wav
```

This matches the structure of your public Drive folder
(`https://drive.google.com/drive/folders/1A_INdaAl-16mMscOpzO-Qcj37rOgfOKE`),
so a direct copy of the folder tree is fine. Note: gdown gets rate-limited on
that folder (~90 files then blocked), so use your original archives / a local
copy if you have one.

**What happens once it's there:** ingest + v1 trial lists (cross-session via
video IDs, open-set dev/test speaker split, real si↔ta cross-lingual pairs),
zero-shot table rerun, P3 arms rerun → benchmark paper table + the proper
LoRA-vs-full verdict.

## 2. Remount NFS on the compute nodes (needs your sudo; ~2 min)

```bash
for n in compute-node-1 compute-node-2 compute-node-3 compute-node-4; do
  ssh $n 'sudo mount /mnt/ricproject3; sudo mount /mnt/ricproject2; sudo mount /mnt/ricproject'
done
```

fstab entries already exist on every node. This is what killed the May
EN-baseline runs (not the 564 bad wavs — the config already uses
`train_list.clean.txt`). Until remounted, all GPU jobs rely on staging copies
under `/home/anuraj/sl_spv_bench/`.

**Unblocks:** EN VoxCeleb baseline training, full-corpus (uncapped) SL
training, and removes the staging step from every future run.

## 3. Decide: commit the new work to git

Currently uncommitted: `tools/ingest_openslr.py`, `tools/zeroshot_eval.py`,
`tools/peft_finetune.py`, `tools/backend_adapt.py`, `tools/extract_p3.py`,
`tools/extract_files.py`, upgraded `tools/sl_dataprep.py`,
`SL_ZEROSHOT_RUNBOOK.md`, this file, the `research_logs/2026-07-03-*.md`
docs, and (updated 2026-07-03 evening) the thesis chapters
(`thesis_chapters/`, `thesis_master.tex` — rebuilt PDF, 18 pp) and paper
drafts (`papers/ieee_spl/` fully rewritten with v0 results; the other three
scaffolds annotated). Say the word and Claude commits them (or commit
yourself).

## 4. Optional / soon

- ~~**3-seed replication** of the winning P3 arm~~ **DONE (2026-07-03):**
  lora and full both ran seeds 42/43/44 — full 1.69±0.06 (si) / 1.41±0.10
  (ta); lora 3.73±0.49 / 3.10±0.61. Tables are in the P3 log §4.1, thesis
  ch. 4, and the IEEE SPL draft.
- **IEEE DataPort SLCeleb page**: consider adding a note/link for the
  Google-Drive rate limit issue if others will download it.
- **SLR52 gender metadata**: Sinhala impostor trials are not gender-controlled
  (no gender labels in SLR52). If you have any speaker metadata for it, place
  it as a CSV (spk_id,gender) and the trial lists regenerate stricter.

## Current research-doc index (one record per improvement)

| Record | File |
|---|---|
| Audit + SOTA roadmap (P0–P6) | `research_logs/2026-07-03-project-audit-sota-roadmap.md` |
| P0 benchmark design | `research_logs/2026-07-03-sl-benchmark-v0-design.md` |
| P1 zero-shot baseline table | `research_logs/2026-07-03-zeroshot-baseline-table.md` |
| P2 backend adaptation (AS-Norm/PLDA/calibration) | `research_logs/2026-07-03-p2-backend-adaptation.md` |
| P3 PEFT fine-tuning | `research_logs/2026-07-03-p3-peft-finetuning.md` |
| How-to-run (P0+P1 pipeline) | `SL_ZEROSHOT_RUNBOOK.md` |
