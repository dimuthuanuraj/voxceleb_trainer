#!/usr/bin/env python3
"""W0.5 -- Register the arbiter front-ends.

The gap
-------
The v1 registry defines two SSL configurations that differ on *two* axes at once:

    ssl_wavlm_lw   layer-weighted, encoder FROZEN        si 2.369 held-out
    ssl_wavlm_ft   last-layer,     encoder FINE-TUNED    si 5.404 held-out

So v1 measured (weighted, frozen) and (last, fine-tuned) but **never the
diagonal** -- layer-weighted *and* fine-tuned. The annual report names that
missing cell as open item #1 and notes that the P3 headline claim currently
underpinning `papers/ieee_spl` rests on it.

Why no model code is needed
---------------------------
`SSLFrontendSpeakerLW` already accepts `ssl_freeze=False` -- it subclasses
`SSLFrontendSpeaker` and its `_extract_ssl_features` has both branches, keeping
the layer weights trainable either way. The trainer already accepts
`--no_ssl_freeze --llrd`. The arbiter is therefore a registry entry, not an
implementation.

What this writes
----------------
Two entries appended to `FRONTENDS` in `experiments/registry.py`:

    ssl_wavlm_lw_ft      WavLM,      layer-weighted + fine-tuned
    ssl_mhubert_lw_ft    mHuBERT-147, layer-weighted + fine-tuned

`experiments/registry.py` is the declared exception to the isolation rule --
Stage F/H entries live there and the harness reads them. `voxceleb_trainer/`
model and loss code is untouched.
"""
from __future__ import annotations

import ast
import datetime
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRW = os.path.dirname(os.path.dirname(HERE))
SL_SPV = os.path.abspath(os.path.join(FRW, "..", "..", ".."))
EXP = os.path.join(SL_SPV, "voxceleb_trainer", "experiments")
REGISTRY = os.path.join(EXP, "registry.py")
OUT = os.path.join(FRW, "03_RESULTS", "W0.5")

ANCHOR = '''    "ssl_mhubert_lw": {'''

ENTRIES = '''    # ----------------------------------------------------------------
    # THE ARBITER RUN  [W0.5, added 2026-09-12]
    #
    # v1 measured layer-weighted+FROZEN (si 2.369) and last-layer+FINE-TUNED
    # (si 5.404) but never both at once, so the two axes were never separated.
    # The annual report of 2026-09-10 names this missing diagonal as open item
    # #1 and notes that the P3 headline claim underpinning papers/ieee_spl
    # rests on it.
    #
    # No new model code: SSLFrontendSpeakerLW subclasses SSLFrontendSpeaker and
    # its _extract_ssl_features already has the unfrozen branch, keeping the 13
    # layer weights trainable either way. Only the flags change.
    #
    # Deviations are inherited verbatim from ssl_wavlm_ft so the comparison is
    # single-factor: same lr, same LLRD decay, same batch size, same memory
    # class. The ONLY difference from ssl_wavlm_ft is reading all 13 hidden
    # states through learned weights instead of the last one.
    # ----------------------------------------------------------------
    "ssl_wavlm_lw_ft": {
        "model": "SSLFrontendSpeakerLW",
        "args": {"nOut": EMBED_DIM},
        "extra_cli": {
            "--ssl_encoder_name": os.path.join(SSL_WEIGHTS_DIR, "wavlm-base-plus"),
        },
        "flags": ["--no_ssl_freeze", "--llrd"],
        "overrides": {"lr": 1e-4, "llrd_decay": 0.9},
        "batch_size": 24,
        "min_gpu_mb": 30000,
        "family": "ssl_layer_weighted_finetuned",
        "note": "THE ARBITER: WavLM layer-weighted AND fine-tuned — the cell v1 "
                "never measured, and the one P3's headline claim rests on",
        "deviations": {
            "lr": "1e-4 with LLRD 0.9 (inherited from ssl_wavlm_ft verbatim so "
                  "the layer-weighting contrast is the single changed factor)",
            "ssl_freeze": "False (as ssl_wavlm_ft)",
        },
    },
    "ssl_mhubert_lw_ft": {
        "model": "SSLFrontendSpeakerLW",
        "args": {"nOut": EMBED_DIM},
        "extra_cli": {
            "--ssl_encoder_name": os.path.join(SSL_WEIGHTS_DIR, "mhubert-147"),
        },
        "flags": ["--no_ssl_freeze", "--llrd"],
        "overrides": {"lr": 1e-4, "llrd_decay": 0.9},
        "batch_size": 24,
        "min_gpu_mb": 30000,
        "family": "ssl_layer_weighted_finetuned",
        "note": "The arbiter on the encoder that won v1 (mHuBERT-147 beat WavLM "
                "on both languages at matched capacity, p = 0.000 / 0.005)",
        "deviations": {
            "lr": "1e-4 with LLRD 0.9 (inherited from ssl_wavlm_ft)",
            "ssl_freeze": "False",
        },
    },
'''


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    report = {"task": "W0.5", "ok": False,
              "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}

    if not os.path.exists(REGISTRY):
        print(f"  MISSING {REGISTRY}")
        json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
        return 1

    src = open(REGISTRY, encoding="utf-8").read()

    if '"ssl_wavlm_lw_ft"' in src:
        print("  already registered")
        report["status"] = "already-registered"
    else:
        if ANCHOR not in src:
            print("  FAILED: could not find the ssl_mhubert_lw anchor in FRONTENDS")
            report["status"] = "NO-ANCHOR"
            json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
            return 1
        backup = REGISTRY + ".pre-W0.5"
        if not os.path.exists(backup):
            shutil.copy2(REGISTRY, backup)
        open(REGISTRY, "w", encoding="utf-8").write(src.replace(ANCHOR, ENTRIES + ANCHOR, 1))
        ast.parse(open(REGISTRY, encoding="utf-8").read())
        print(f"  registered 2 arbiter front-ends (backup: {os.path.basename(backup)})")
        report["status"] = "registered"

    # ------------------------------------------------------------- verify
    print("\n  verifying the registry builds a valid experiment...")
    probe = os.path.join(OUT, "_probe.py")
    open(probe, "w").write(f'''
import json, sys, os
sys.path.insert(0, {EXP!r})
import registry as r

res = {{}}
res["registered"] = [k for k in r.FRONTENDS if k.endswith("_lw_ft")]

# The arbiter must differ from ssl_wavlm_ft on exactly one thing: the model.
a, b = r.FRONTENDS["ssl_wavlm_lw_ft"], r.FRONTENDS["ssl_wavlm_ft"]
diff = sorted(k for k in set(a) | set(b)
              if k not in ("note", "family", "deviations") and a.get(k) != b.get(k))
res["differs_from_ssl_wavlm_ft_on"] = diff
res["single_factor"] = diff == ["model"]

# ...and from ssl_wavlm_lw on exactly the freezing/optimiser package.
c = r.FRONTENDS["ssl_wavlm_lw"]
res["differs_from_ssl_wavlm_lw_on"] = sorted(
    k for k in set(a) | set(c)
    if k not in ("note", "family", "deviations") and a.get(k) != c.get(k))

# Build a real experiment and emit its argv -- the actual test.
try:
    exps = r.stage_f(conditions=("si",))
    res["stage_f_builds"] = True
except Exception as exc:
    res["stage_f_builds"] = False
    res["stage_f_error"] = repr(exc)

try:
    e = r.Experiment(exp_id="F_ssl_wavlm_lw_ft_aamsoftmax_si_s42", stage="F",
                     arch_key="ssl_wavlm_lw_ft", loss_key="aamsoftmax",
                     condition_key="si", seed=42)
    argv = e.argv()
    res["argv_ok"] = True
    res["argv"] = argv if isinstance(argv, list) else str(argv)
    res["min_gpu_mb"] = e.min_gpu_mb()
    joined = " ".join(map(str, argv))
    res["has_no_ssl_freeze"] = "--no_ssl_freeze" in joined
    res["has_llrd"] = "--llrd" in joined
    res["model_is_lw"] = "SSLFrontendSpeakerLW" in joined
except Exception as exc:
    res["argv_ok"] = False
    res["argv_error"] = repr(exc)

print(json.dumps(res, indent=2)[:4000])
open({os.path.join(OUT, "probe.json")!r}, "w").write(json.dumps(res, indent=2))
''')
    subprocess.call([sys.executable, probe])

    ppath = os.path.join(OUT, "probe.json")
    p = json.load(open(ppath)) if os.path.exists(ppath) else {}
    report["probe"] = p

    ok = (len(p.get("registered", [])) == 2
          and p.get("argv_ok")
          and p.get("has_no_ssl_freeze")
          and p.get("has_llrd")
          and p.get("model_is_lw"))

    if ok:
        print(f"\n  arbiter argv builds; model={p.get('model_is_lw')} "
              f"no_ssl_freeze={p.get('has_no_ssl_freeze')} llrd={p.get('has_llrd')}")
        print(f"  min_gpu_mb = {p.get('min_gpu_mb')}")
        if p.get("single_factor"):
            print("  single-factor vs ssl_wavlm_ft: only `model` differs -- the contrast is clean")
        else:
            print(f"  NOTE differs from ssl_wavlm_ft on: {p.get('differs_from_ssl_wavlm_ft_on')}")
    else:
        print("\n  VERIFICATION FAILED -- inspect probe.json")

    report["ok"] = bool(ok)
    json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
    print(f"\n  -> {os.path.join(OUT, 'result.json')}  ok={report['ok']}")
    if ok:
        print(f"\n  The arbiter needs {p.get('min_gpu_mb')} MiB. Largest live GPU today is"
              " 14,914 MiB.\n  B1/B2 stay blocked on W0.4 until an A40 or A10 returns.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
