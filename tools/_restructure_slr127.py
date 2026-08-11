#!/usr/bin/env python3
"""One-off: re-key the slr127 tree after the prefix-identity correction.

The decoded audio is unchanged -- only which speaker directory a file belongs to
changes -- so this moves files instead of re-decoding 89k wavs.

    old  wav/ta/<number>/<PREFIX>/<full_stem>.wav
    new  wav/ta/<PREFIX>_<number>/<full_stem>/00001.wav

Then it regenerates metadata/*.csv, metadata.json and lists/ from the new tree.
Running data/slr127_tamil/prepare.py from scratch produces the same result; this
is the fast path.
"""
import csv
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DS = os.path.join(REPO, "data", "slr127_tamil")
sys.path.insert(0, os.path.join(REPO, "data", "_common"))
import slprep  # noqa: E402

WAV = os.path.join(DS, "wav")
OLD_CSV = os.path.join(DS, "metadata", "utterances.csv")


def main():
    with open(OLD_CSV, encoding="utf-8") as fh:
        old = list(csv.DictReader(fh))
    print(f"[restructure] {len(old)} rows in the old manifest")

    rows, moved, missing = [], 0, 0
    for r in old:
        # old path: ta/<number>/<PREFIX>/<stem>.wav
        parts = r["path"].split("/")
        if len(parts) != 4:
            missing += 1
            continue
        lang, number, prefix, fname = parts
        stem = os.path.splitext(fname)[0]
        new_spk = f"{prefix}_{number}"
        new_rel = f"{lang}/{new_spk}/{stem}/00001.wav"
        src = os.path.join(WAV, r["path"])
        dst = os.path.join(WAV, new_rel)
        if not os.path.exists(dst):
            if not os.path.exists(src):
                missing += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved += 1
        rows.append({
            "path": new_rel, "lang": lang, "spk_id": new_spk,
            "orig_spk_id": number, "session_id": stem, "utt_id": "00001",
            "gender": r["gender"], "duration_s": float(r["duration_s"]),
            "orig_sr": int(r["orig_sr"]), "orig_channels": int(r["orig_channels"]),
            "src_path": r["src_path"],
        })
        if moved and moved % 20000 == 0:
            print(f"  moved {moved}", flush=True)

    print(f"[restructure] moved {moved}, missing {missing}, kept {len(rows)}")
    rows.sort(key=lambda x: x["path"])

    # drop the now-empty old speaker directories
    removed = 0
    for d in sorted(os.listdir(os.path.join(WAV, "ta"))):
        p = os.path.join(WAV, "ta", d)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)
            removed += 1
    print(f"[restructure] removed {removed} empty directories")

    spk_rows = slprep.write_tables(DS, rows)
    summary = slprep.summarise(rows, spk_rows)
    meta_path = os.path.join(DS, "metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta["statistics"] = summary
    meta["session_semantics"] = {
        "derivation": "one session directory per source utterance",
        "true_multi_session": False,
        "note": "No session metadata exists in this corpus. An earlier version of "
                "prepare.py read the collection prefix as a session; the Layer-3 "
                "audit disproved that. Each utterance is an independent recording, "
                "so target pairs never share a recording, but they do share a "
                "sitting and channel -- as in SLR52 and SLR65.",
    }
    meta["prefix_identity_finding"] = {
        "date": "2026-08-10",
        "evidence": "data/_qc/slr127_prefix_identity.json",
        "same_number_cross_prefix_cosine": 0.269,
        "different_speaker_baseline_cosine": 0.319,
        "same_speaker_within_prefix_cosine": 0.961,
        "numbers_reused_across_prefixes": 107,
        "conclusion": "<PREFIX>_<number> is the speaker key; the same number under "
                      "two prefixes is two different people.",
        "impact": "More speakers than the number field alone suggests, and NO "
                  "genuine cross-session trials.",
    }
    meta["caveats"] = [
        "No gender labels, so same-gender impostor sampling is unavailable.",
        "NO genuine cross-session trials -- the prefix is a numbering scheme, not "
        "a session. Absolute EER is optimistic, as for SLR52 and SLR65.",
        "Indian Tamil, so it cannot support a Sri Lankan Tamil claim.",
    ]
    meta["suitable_for"] = ["training", "in-domain evaluation"]
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"[restructure] speakers={summary['speakers_total']} "
          f"utts={summary['utterances_total']} hours={summary['hours_total']}")
    slprep.run_dataprep(DS, ["ta"], seed=42, target_pairs=3000, impostor_ratio=3)
    print("[restructure] done")


if __name__ == "__main__":
    main()
