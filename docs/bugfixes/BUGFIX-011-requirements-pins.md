# BUGFIX-011 — `requirements.txt` had loose pins (`torch>=1.7.0`) that mismatch the actual training environment and allow accidental breaking-major upgrades

| Field | Value |
|---|---|
| **ID** | BUGFIX-011 |
| **Severity** | Medium (no active runtime crash, but `pip install -r requirements.txt` on a clean machine can land an inconsistent or breaking dependency set) |
| **Component** | Dependency manifest |
| **Files touched** | [requirements.txt](../../requirements.txt) |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #10 |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

---

## 1. Problem

### 1.1 The pre-fix manifest
```
torch>=1.7.0
torchaudio>=0.7.0
asteroid_filterbanks==0.4.0
numpy
scipy
scikit-learn
tqdm
pyyaml
soundfile
```

Two distinct problems:

1. **`torch>=1.7.0` is too loose.** It permits PyTorch versions
   spanning ≈ 5 years of API churn. A clean install on a current
   machine resolves to PyTorch 2.9 (which the team is actually
   using per [`research_logs/2025-10-20.md`](../../research_logs/2025-10-20.md));
   a clean install with `pip install --constraint <old>` could land
   PyTorch 1.7, which lacks several APIs the code relies on.

2. **Six unconstrained dependencies** (`numpy`, `scipy`,
   `scikit-learn`, `tqdm`, `pyyaml`, `soundfile`). Each is one
   `pip install -U` away from a breaking major bump that the
   project hasn't tested against. NumPy 2.0 specifically already
   bit this repo once — the [`research_logs/2025-10-20.md`](../../research_logs/2025-10-20.md)
   entry documents fixing `numpy.float` AttributeErrors after NumPy
   migrated to its 2.x semantics.

### 1.2 Audit — what does the code actually use?
Honest accounting of the **lowest** versions that satisfy the API surface:

| Feature | First available in | Why we use it |
|---|---|---|
| `torch.cuda.amp.GradScaler(enabled=...)` (kwarg form) | 1.6 | `SpeakerNet_performance_updated.py:104`, `SpeakerNet_distillation.py:162` — configurable mixed-precision |
| `Module.register_buffer(name, tensor, persistent=False)` | 1.6 | Closed in [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md) for the SincConv frontend; also in [`utils.py:29`](../../utils.py), [`models/RawNetBasicBlock.py:18`](../../models/RawNetBasicBlock.py) |
| `Optimizer.zero_grad(set_to_none=True)` | 1.7 | `SpeakerNet_performance_updated.py:139,148` — memory-efficient gradient reset |
| `torch.inference_mode()` | 1.9 | `SpeakerNet_performance_updated.py:232,283` — faster than `no_grad()` for eval |
| `torchaudio.transforms.MelSpectrogram(..., window_fn=torch.hamming_window)` | torchaudio 0.7 | Every mel-based model |
| `numpy.random.randn`, `numpy.pad`, `numpy.concatenate`, `numpy.linspace`, ... | numpy 1.x | Everywhere |
| `signal.resample_poly`, `signal.fftconvolve`, `signal.convolve` | scipy 1.7 | `DatasetLoader.py:90`, `DatasetLoader_performance_updated.py:225` — added in [BUGFIX-005](BUGFIX-005-sample-rate-hardcoded.md) and `AugmentWAV.reverberate` |
| `sklearn.metrics.roc_curve` | sklearn 0.18 | `tuneThreshold.py:15` |
| `soundfile.read(filename, dtype='float32')` | soundfile 0.9 | Every loader |

So the **strict** minimum is `torch>=1.9` + `torchaudio>=0.9`.
The audit report in `SL_LANGUAGE_SPV_ANALYSIS.md` claimed the code
uses `torch.amp.autocast('cuda', ...)` (the device-typed
PyTorch 2.1+ form). That claim was **inaccurate** — the actual
code uses the older `torch.cuda.amp.autocast()` form throughout.
The bug-report's stated reason for pinning ≥2.1 doesn't hold up
under audit; this is noted honestly here.

### 1.3 Why we still pin `torch>=2.1,<3` despite §1.2

Even with the inaccurate report reason, `2.1` is still the right
floor. Three independent reasons:

1. **The team's actual training environment is PyTorch 2.9.0+cu128**
   ([`research_logs/2025-10-20.md`](../../research_logs/2025-10-20.md) Environment Setup section).
   Pinning to anything less than 2.x is asking for a "works on
   teammate's machine, breaks on mine" failure mode.
2. **NumPy 2.x compatibility window.** PyTorch <2.3 has known
   issues interoperating with NumPy 2.x; the project has already
   migrated to NumPy 2.x semantics (per `research_logs/2025-10-20.md`),
   so the floor should at minimum match. 2.1 is conservative
   enough to allow CUDA-12-compatible builds.
3. **API drift insurance.** When `torch.cuda.amp.autocast` is
   eventually migrated to `torch.amp.autocast('cuda', ...)` (a
   small follow-up the project will want at some point), the
   floor will need to be 2.1 anyway. Pinning now avoids
   re-touching this file then.

### 1.4 Why ceilings matter

PyTorch's API stability across major versions has historically
been tight but not perfect. A 3.0 release is plausible within
the next two years and would, by SemVer convention, contain
breaking changes. NumPy 2.0 is the cautionary tale: a single
unconstrained `numpy` in `requirements.txt` was enough to break
clean installs in the wild. Bounded ceilings (`<3`, `<2`, `<7`)
let `pip install` succeed reproducibly while still allowing
minor / patch updates.

---

## 2. Fix applied

### 2.1 Diff

```diff
-torch>=1.7.0
-torchaudio>=0.7.0
-asteroid_filterbanks==0.4.0
-numpy
-scipy
-scikit-learn
-tqdm
-pyyaml
-soundfile
+# (full file replaced — see §2.2 below for the new structure)
+torch>=2.1,<3
+torchaudio>=2.1,<3
+asteroid_filterbanks==0.4.0
+scipy>=1.7,<2
+soundfile>=0.10,<1
+numpy>=1.20,<3
+scikit-learn>=1.0,<2
+tqdm>=4.0,<5
+pyyaml>=5.0,<7
+# tensorboard>=2.0,<3     # optional
+# matplotlib>=3.0,<4      # optional
```

Plus a structured set of comments grouping the pins by purpose
(PyTorch ecosystem / audio / numerical / optional). The full
formatted file is in [requirements.txt](../../requirements.txt).

### 2.2 Pin rationale per dependency

| Dependency | Pin | Why |
|---|---|---|
| `torch>=2.1,<3` | major-bounded | §1.3 — matches actual env, NumPy-2 safe, future-proofs the autocast migration |
| `torchaudio>=2.1,<3` | major-bounded | Must match torch's major version per the PyTorch release-pairing matrix; same floor as torch |
| `asteroid_filterbanks==0.4.0` | exact | Pre-existing pin — `RawNet3.py:30` constructs `ParamSincFB` with a 0.4.0-specific kwarg signature |
| `scipy>=1.7,<2` | major-bounded | 1.7 has stable `signal.resample_poly`; `<2` because scipy 2.0 is plausibly breaking |
| `soundfile>=0.10,<1` | major-bounded | 0.10 is when `dtype='float32'` reading stabilised; pre-1.0 means major bumps are still possible |
| `numpy>=1.20,<3` | major-bounded | 1.20 begins the `numpy.float` deprecation series which the project has fixed; `<3` future-proofs against the next breaking major |
| `scikit-learn>=1.0,<2` | major-bounded | Only `metrics.roc_curve` is used; very stable |
| `tqdm>=4.0,<5` | major-bounded | Only progress bars; stable since 4.0 |
| `pyyaml>=5.0,<7` | conservative major-bounded | `yaml.load(..., Loader=yaml.FullLoader)` requires 5.1+; `<7` permits 6.x (current as of writing) |
| `tensorboard` / `matplotlib` | commented-out | Truly optional — imported via try/except in `trainSpeakerNet*.py:20-32` |

### 2.3 The commented-out optional block

`tensorboard` and `matplotlib` are imported with `try/except` in
all three trainers (`trainSpeakerNet.py:20-32` and the parallel
spots in the `_performance_updated` and `_distillation` scripts).
If absent at runtime, the trainer prints a polite "install X to
enable Y" line and proceeds without that feature.

Three plausible designs for these:

1. **Always include them** (uncommented in requirements). Pro:
   `pip install -r requirements.txt` gives the fully-featured
   environment. Con: forces dependencies that the trainer
   doesn't actually need to run.
2. **Always exclude them** (drop from requirements entirely).
   Pro: minimum install size. Con: every new user discovers them
   one trainer-run at a time.
3. **Comment them out with a one-line nudge** (this fix).
   Pro: the trainer's runtime "install tensorboard to enable
   logging" message lands directly on a copy-pasteable line in
   `requirements.txt`. Con: requires an extra step (uncomment +
   `pip install -r`) for the full experience.

The project's existing trainer messages already say
"Please run `pip install tensorboard`" — they expect the user to
add these manually. The commented form matches that contract
without surprising anyone, and keeps the trainer's silent-fallback
behaviour the default.

---

## 3. Verification

### 3.1 Static — syntax parse
```bash
$ python3 -c "
import re
with open('requirements.txt') as f:
    for line_no, raw in enumerate(f, 1):
        line = raw.split('#', 1)[0].strip()
        if not line: continue
        m = re.match(r'^([A-Za-z0-9_\-\.]+)\s*(.*)$', line)
        assert m, f'line {line_no}: {raw!r}'
        name, rest = m.group(1), m.group(2).strip()
        if rest:
            for spec in rest.split(','):
                assert re.match(r'^\s*(==|>=|<=|!=|~=|>|<)\s*[0-9]', spec), \
                    f'line {line_no}: malformed spec {spec!r}'
print('OK')
"
OK
```

Nine active lines, all valid PEP 508 form. Two commented optional
lines, ignored by the parser.

### 3.2 Functional — clean install in a fresh venv
The recommended in-env check (not run on this analysis machine
which has no pip cache):
```bash
python3 -m venv /tmp/sl-venv
source /tmp/sl-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "
import torch, torchaudio, numpy, scipy, sklearn, soundfile, tqdm, yaml, asteroid_filterbanks
print(f'torch={torch.__version__}, torchaudio={torchaudio.__version__}')
print(f'numpy={numpy.__version__}, scipy={scipy.__version__}, sklearn={sklearn.__version__}')
print(f'soundfile={soundfile.__version__}, asteroid={asteroid_filterbanks.__version__}')
"
```
Expected: every line resolves and prints versions inside the
pinned ranges. Torch should land at 2.x (whatever current
release is); none of `numpy` / `scipy` / `sklearn` should jump
to a major-bump version.

### 3.3 Functional — does the existing checkpoint still load?
The single most important runtime invariant after a dependency
update is that the existing best checkpoints (MLP-Mixer V2 at
10.32% EER and LSTM+AE teacher at 9.68% EER) still load:
```bash
python trainSpeakerNet.py --eval \
    --config configs/mlp_mixer_distillation_v2.yaml \
    --initial_model exps/mlp_mixer_distillation_v2/model/model000000060.model
```
Expected: EER reproduces (10.75% on that specific checkpoint per
[`research_logs/2025-12-30-31-experimental-results-analysis.md`](../../research_logs/2025-12-30-31-experimental-results-analysis.md)).
A mismatch would indicate a torch upgrade introduced a numerical
divergence; investigate before continuing work.

---

## 4. Backward compatibility

| Consumer | Effect |
|---|---|
| The team's current training env (PyTorch 2.9.0+cu128) | ✅ Bit-identical — `>=2.1,<3` matches 2.9 |
| A new contributor running `pip install -r requirements.txt` on a clean machine | ✅ Now resolves to a consistent recent PyTorch 2.x set instead of either an out-of-date 1.7 install (would crash on `inference_mode`) or an unconstrained latest (could break on a breaking major) |
| A CI / Docker image rebuild | ✅ Pinned upper bounds prevent the kind of "I rebuilt the image and now training broke" surprise the project hit on NumPy 2.0 |
| Anyone explicitly running PyTorch 1.x | ❌ Their setup will now fail at `pip install`. **This is desired** — `torch.inference_mode` would have crashed at first use anyway; a loud install-time failure is strictly better than a silent runtime failure. |
| Anyone using a custom `torch` build (e.g., nightly) | ⚠️ Nightlies use a `dev` suffix; `pip` treats them as pre-releases. If working from a nightly, run `pip install --pre -r requirements.txt`. Otherwise no change. |

### 4.1 No checkpoint / config migration required
Pinned dependency versions do not touch checkpoint format, config
schema, or any persisted on-disk artifact. Existing `exps/`
directories and `configs/` YAMLs continue to work.

---

## 5. Things this fix does NOT change

| Item | Why deferred |
|---|---|
| Adding `pyproject.toml` / `setup.py` for proper package metadata | The repo is structured as a training-script collection, not an importable package. A proper packaging pass is a larger refactor. |
| Generating a `requirements.lock` (full transitive lockfile via pip-tools / poetry) | A real lockfile gives bit-exact reproducibility but requires regenerating on every dep change. For an actively-developed research project that wants minor / patch updates, major-bounded pins are a better fit. Worth revisiting if reproducibility ever becomes paper-grade. |
| Migrating from `torch.cuda.amp.autocast()` to `torch.amp.autocast('cuda', ...)` | Independent code change. The audit at §1.2 confirms the current code uses the old form; migration is mechanical (one import + N usage sites) but orthogonal to the dependency pinning. |
| Pinning `tensorboard` / `matplotlib` (optional) | Truly optional; the trainer falls back gracefully. Listed commented-out with version bounds so a user who wants them gets a sane install. |
| Splitting into `requirements-dev.txt` / `requirements-test.txt` | The repo has no formal test suite or dev-only tooling. A split is unnecessary now and trivially added later if it becomes one. |

---

## 6. Rollback

Restore the original 9-line file. There is no scenario in which
rollback is correct: the pre-fix `torch>=1.7.0` floor permits an
install set that will immediately crash at `torch.inference_mode()`
on the first eval pass, and the unconstrained `numpy` line was
already shown to cause concrete breakage in October 2025.

---

## 7. Closes / related

| Item | Status |
|---|---|
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 #10 (loose pins) | ✅ Closed by this document |
| §4.1 #1–#9 (Critical) | ✅ BUGFIX-001..010 (§4.1 fully closed) |
| §4.2 #11 — empty `analyze_nan_debug.py` / `NaN_DEBUGGING_GUIDE.md` placeholders | ✅ [BUGFIX-012](BUGFIX-012-fill-nan-debug-placeholders.md) (filled with real content) |
| §4.2 #13 — configs hard-code `/mnt/ricproject*/` paths | ✅ [BUGFIX-013](BUGFIX-013-portable-config-paths.md) (env-var substitution) |
| §4.2 #14 — `n_mels` ignored in `ResNetSE34L.py`/`VGGVox.py` | ✅ [BUGFIX-014](BUGFIX-014-honour-n-mels-in-vggvox.md) |
| §4.2 #15 — `RawNet3.py` debug print + in-place mutation | ✅ [BUGFIX-015](BUGFIX-015-rawnet3-debug-print-and-inplace.md) |
| §4.2 #16 — Augmentation hard-codes 5 fixed choices | ✅ [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) |
| §4.2 #17 — No deterministic mode toggle | ✅ [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) |
| §4.2 #18 — `evaluateFromList` loads all features into rank-0 dict | ✅ [BUGFIX-018](BUGFIX-018-streaming-evaluation.md) |
| §4.2 #19 — `torch.load` without `weights_only=True` | ✅ [BUGFIX-019](BUGFIX-019-torch-load-weights-only.md) |
| §4.2 #20 — EER definition differs from common `(fpr+fnr)/2` | ✅ [BUGFIX-020](BUGFIX-020-eer-definition-disclosure.md) |
| §4.3 #21 — `pdb` imports left in production | ✅ [BUGFIX-021](BUGFIX-021-remove-pdb-imports.md) |
| §4.3 #22 — Inconsistent variable casing | ✅ [BUGFIX-022](BUGFIX-022-camel-snake-case-aliases.md) |
| §4.3 #23 — `dataprep.py` MD5 mismatch raises `Warning` | ✅ [BUGFIX-023](BUGFIX-023-md5-mismatch-proper-error.md) |
| §4.3 #24 — Performance scripts overlap | ✅ [BUGFIX-024](BUGFIX-024-consolidate-performance-scripts.md) |
| §4.3 #25 — Each model duplicates PreEmphasis / mel / InstanceNorm | ✅ [BUGFIX-025](BUGFIX-025-shared-audio-frontend.md) |
| §4.3 #26 — `exps/` cleanup policy | ✅ [BUGFIX-026](BUGFIX-026-exps-cleanup-policy.md) |
| §4.2 #18 — `evaluateFromList` loads all features into rank-0 dict | ⬜ Open, scaling concern for >500k-pair test lists |
| §4.2 #19 — `torch.load` without `weights_only=True` | ⬜ Open, security hardening for shared checkpoints |
| Polish: `torch.amp.autocast('cuda', ...)` migration | ⬜ Open, mechanical, follow-up |

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #10.
- **Honest correction to the original report:** §1.2 of this doc
  documents that the report's stated reason (code uses
  `torch.amp.autocast('cuda', ...)`) is inaccurate — the code uses
  the older `torch.cuda.amp.autocast()` form. The fix is still
  correct, for the three independent reasons in §1.3.
- **PyTorch ↔ torchaudio release pairing:** https://pytorch.org/audio/stable/installation.html
- **NumPy 2.0 migration notes:** https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
- **PEP 440 version specifiers:** https://peps.python.org/pep-0440/#version-specifiers
- **PEP 508 dependency specification:** https://peps.python.org/pep-0508/
- **Project's prior NumPy-2 fix:** `research_logs/2025-10-20.md`.
