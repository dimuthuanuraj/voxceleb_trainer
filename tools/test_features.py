#!/usr/bin/env python3
"""
SL-SPV feature smoke tests — exercise every shipped feature on
synthetic data, with no SL corpus required. Each test asserts a
specific behavioural property and reports PASS/FAIL. Designed to
run in under 60 seconds on CPU.

Usage:
    python tools/test_features.py
    python tools/test_features.py --verbose
    python tools/test_features.py --feature FEATURE-006     # run just one

Exit code 0 if all tests pass, 1 if any test fails.
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile
import traceback
from pathlib import Path

# Allow running from the repo root (parent of tools/)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"


# ----- Individual tests -------------------------------------------------------

def test_feature_002_as_norm():
    """FEATURE-002 — AS-Norm: cohort extraction + per-file stats + apply."""
    import numpy as np, torch
    from score_norm import compute_file_cohort_stats, apply_as_norm

    cohort = torch.randn(20, 5, 32)   # 20 cohort speakers, 5 num_eval, 32 dim
    file_feats = {f"f{i}": torch.randn(5, 32) for i in range(8)}

    stats = compute_file_cohort_stats(
        file_iter=iter(file_feats.items()),
        cohort_feats=cohort, top_k=10, normalize=True, device='cpu',
    )
    assert len(stats) == 8
    for path, (mu, sig) in stats.items():
        assert isinstance(mu, float) and isinstance(sig, float)
        assert sig > 0, f"sigma must be positive (eps-floored), got {sig}"

    # apply_as_norm
    raw_scores = [0.5, -0.3, 0.7]
    trials = ["f0 f1", "f2 f3", "f4 f5"]
    normed = apply_as_norm(raw_scores, trials, stats)
    assert len(normed) == 3 and all(isinstance(s, float) for s in normed)
    return "AS-Norm: cohort stats + apply produce finite normalised scores."


def test_feature_003_per_lang_eval():
    """FEATURE-003 — per-language test-list parsing."""
    sys.path.insert(0, str(REPO_ROOT))
    # Import the helper that lives in trainSpeakerNet.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tsn", REPO_ROOT / "trainSpeakerNet.py"
    )
    # trainSpeakerNet.py runs argparse at import; capture stdout/stderr.
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        # Override sys.argv to make argparse happy with no flags.
        old_argv = sys.argv[:]
        sys.argv = ["trainSpeakerNet.py"]
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.argv = old_argv

    # CLI string form
    pairs = mod._parse_per_lang_test_lists("si:/a.txt,ta:/b.txt,cs:/c.txt")
    assert pairs == [("si", "/a.txt"), ("ta", "/b.txt"), ("cs", "/c.txt")]

    # YAML dict form
    pairs2 = mod._parse_per_lang_test_lists({"si": "/x.txt", "ta": "/y.txt"})
    assert ("si", "/x.txt") in pairs2 and ("ta", "/y.txt") in pairs2

    # Empty
    assert mod._parse_per_lang_test_lists("") == []
    assert mod._parse_per_lang_test_lists(None) == []
    return "per_lang_test_lists: parses CLI string + YAML dict + empty cleanly."


def test_feature_004_finetune_freeze():
    """FEATURE-004 — alias parse + freeze succeeds + param count conservation.

    Note: 'frontend' (mel torchfb + instancenorm) has NO trainable params on
    ECAPA-TDNN — torchfb is a MelSpectrogram with buffers only, and
    InstanceNorm1d defaults to affine=False. So freezing 'frontend' is a
    no-op on trainable count by design. We test with 'backbone' (__S__),
    which freezes the entire ECAPA encoder, to assert reduction works.
    """
    import torch
    from SpeakerNet import _parse_finetune_freeze, _apply_finetune_freeze
    from models.ECAPA_TDNN import MainModel
    import torch.nn as nn

    inner = MainModel(nOut=64, channels=128)
    class _S(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.__S__ = s
            self.__L__ = nn.Linear(64, 10)
    class _Wrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = _S(inner)
    wrap = _Wrap(inner)
    n_tensors_total = sum(1 for _ in wrap.parameters())

    # 1) 'frontend' alias parses and runs (n_frozen may be 0 — expected).
    specs = _parse_finetune_freeze("frontend")
    assert "module.__S__.torchfb" in specs, "alias resolution"
    n_train, n_frozen, hits = _apply_finetune_freeze(wrap, specs)
    # _apply_finetune_freeze returns counts of parameter TENSORS, not elements.
    assert n_train + n_frozen == n_tensors_total, \
        f"tensor count conservation: {n_train}+{n_frozen} != {n_tensors_total}"

    # 2) 'backbone' freezes the entire __S__, must reduce trainable.
    for p in wrap.parameters():       # reset requires_grad
        p.requires_grad = True
    specs2 = _parse_finetune_freeze("backbone")
    n_train2, n_frozen2, hits2 = _apply_finetune_freeze(wrap, specs2)
    assert n_frozen2 > 0, "freezing backbone must reduce trainable tensors"
    assert n_train2 + n_frozen2 == n_tensors_total
    return (f"frontend alias: hits={hits} (no-op by design); "
            f"backbone: {n_frozen2} tensors frozen")


def test_feature_005_llrd():
    """FEATURE-005 — LLRD param groups have monotonically decaying lr."""
    import re, torch.nn as nn
    from SpeakerNet import _build_llrd_param_groups, _resolve_llrd_pattern
    from models.ECAPA_TDNN import MainModel

    inner = MainModel(nOut=64, channels=128)
    class _S(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.__S__ = s
            self.__L__ = nn.Linear(64, 10)
    class _Wrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = _S(inner)
    wrap = _Wrap(inner)

    layer_re = _resolve_llrd_pattern("ecapa")
    groups, max_n = _build_llrd_param_groups(
        wrap, base_lr=1e-3, decay=0.9, layer_re=layer_re,
    )
    assert groups is not None, "ecapa pattern must catch ECAPA layers"
    lrs = [g['lr'] for g in groups]
    assert lrs == sorted(lrs, reverse=True), \
        f"lrs must be monotonically decreasing with depth; got {lrs}"
    return f"LLRD ecapa: {len(groups)} buckets, lrs={[f'{lr:.2e}' for lr in lrs]}"


def test_feature_006_ecapa_tdnn():
    """FEATURE-006 — ECAPA-TDNN forward produces correct shape + finite."""
    import torch
    from models.ECAPA_TDNN import MainModel
    m = MainModel(nOut=192, channels=512).eval()  # small for speed
    with torch.no_grad():
        y = m(torch.randn(2, 16000))
    assert y.shape == (2, 192), f"expected [2,192], got {y.shape}"
    assert torch.isfinite(y).all(), "forward output must be finite"
    n_params = sum(p.numel() for p in m.parameters())
    return f"ECAPA forward: out={tuple(y.shape)}, params={n_params/1e6:.2f}M"


def test_feature_007_lang_aux():
    """FEATURE-007 — LangAuxHead + lookup loader produces a valid loss."""
    import tempfile, torch
    from SpeakerNet import LangAuxHead, _load_lang_lookup

    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        for spk in range(10):
            f.write(f"{spk} {spk % 3}\n")
        lookup_path = f.name
    try:
        table, n_valid = _load_lang_lookup(lookup_path, n_classes_spk=10,
                                           num_lang_classes=3)
        assert n_valid == 10
        assert table.shape == (10,)
        assert table[5].item() == 5 % 3

        head = LangAuxHead(embedding_dim=16, num_classes=3)
        x = torch.randn(8, 16)
        label = torch.tensor([0, 1, 2, 0, 1, 2, -1, 0])  # one ignored
        loss, prec = head(x, label)
        assert torch.isfinite(loss).item()
        return f"lang_aux: lookup loaded {n_valid}/10, loss={loss.item():.4f}"
    finally:
        os.unlink(lookup_path)


def test_feature_008_dann_grl():
    """FEATURE-008 — Gradient Reversal Layer gives -λ·grad."""
    import torch
    from SpeakerNet import GradientReversalFn

    x = torch.randn(4, 8, requires_grad=True)
    x.sum().backward(retain_graph=True)
    g_base = x.grad.clone()
    x.grad.zero_()

    GradientReversalFn.apply(x, 0.5).sum().backward()
    g_grl = x.grad.clone()
    assert torch.allclose(g_grl, -0.5 * g_base), \
        f"GRL gradient must be -0.5*baseline; got {g_grl[0,0]} vs {-0.5*g_base[0,0]}"
    return "GRL: backward gradient is exactly -λ × baseline."


def test_feature_010_plda():
    """FEATURE-010 — TwoCovPLDA fit + score + save/load on synthetic."""
    import numpy as np, tempfile
    from plda import TwoCovPLDA

    np.random.seed(0)
    n_spk, n_utt, d = 200, 8, 96   # smaller than the design-doc smoke test
    spk_means = np.random.randn(n_spk, d) * 1.5
    embs, labels = [], []
    for s in range(n_spk):
        embs.append(spk_means[s] + 0.3 * np.random.randn(n_utt, d))
        labels.extend([s] * n_utt)
    X = np.concatenate(embs, axis=0)
    y = np.array(labels)

    plda = TwoCovPLDA(lda_dim=80).fit(X, y)
    rng = np.random.RandomState(1)
    same = [plda.score(X[s*n_utt + i], X[s*n_utt + j])
            for s in rng.choice(n_spk, 50)
            for i, j in [rng.choice(n_utt, 2, replace=False)]]
    diff = [plda.score(X[a*n_utt + rng.randint(n_utt)],
                       X[b*n_utt + rng.randint(n_utt)])
            for _ in range(50)
            for a, b in [rng.choice(n_spk, 2, replace=False)]]
    same_m, diff_m = float(np.mean(same)), float(np.mean(diff))
    assert same_m > diff_m, f"same-spk mean ({same_m:.3f}) must exceed diff ({diff_m:.3f})"

    # Save/load round-trip
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        plda.save(path)
        plda2 = TwoCovPLDA.load(path)
        s1 = plda.score(X[0], X[1]); s2 = plda2.score(X[0], X[1])
        assert abs(s1 - s2) < 1e-9
    finally:
        os.unlink(path)
    return f"PLDA: same={same_m:.3f}, diff={diff_m:.3f}, save/load round-trip OK"


def test_feature_011_dataprep():
    """FEATURE-011 — sl_dataprep.py on synthetic 100-speaker tree."""
    import subprocess, tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        # 60 si + 40 ta, 10 utts each
        for lang, n in [("si", 60), ("ta", 40)]:
            for spk in range(n):
                sd = root / lang / f"{lang}_spk{spk:03d}"
                sd.mkdir(parents=True)
                for u in range(10):
                    (sd / f"utt{u:02d}.wav").write_bytes(b"\x00" * 32)
        out = td / "lists"
        rc = subprocess.run(
            ["python", "tools/sl_dataprep.py",
             "--corpus_root", str(root),
             "--out_dir", str(out),
             "--langs", "si", "ta",
             "--test_frac", "0.2",
             "--target_pairs_per_lang", "100",
             "--cohort_size", "20",
             "--seed", "42"],
            capture_output=True, text=True,
        )
        assert rc.returncode == 0, rc.stderr
        expected = ["train_list.txt", "test_list.txt", "test_list_si.txt",
                    "test_list_ta.txt", "test_list_cs.txt",
                    "spk_lang_lookup.txt", "asnorm_cohort.txt",
                    "plda_train_list.txt", "speakers.csv"]
        present = sorted(p.name for p in out.iterdir())
        for e in expected:
            assert e in present, f"missing output file: {e}"
        # speakers.csv must have 100 rows + header
        n_spk = sum(1 for _ in open(out / "speakers.csv")) - 1
        assert n_spk == 100, f"expected 100 speakers, got {n_spk}"
    return "sl_dataprep: 100 speakers, 9 output files generated."


def test_configs_parse():
    """All 12 sl_*.yaml configs parse cleanly and have expected feature flags."""
    import glob, yaml
    os.environ.setdefault("SL_SPV_DATA_ROOT", "/tmp/dummy")
    n_ok = 0
    for path in sorted(glob.glob("configs/sl_*.yaml")):
        cfg = yaml.safe_load(open(path))
        for flag in ("finetune", "llrd", "lang_aux", "as_norm", "plda"):
            assert flag in cfg, f"{path}: missing key {flag}"
            assert isinstance(cfg[flag], bool), f"{path}: {flag} must be bool"
        n_ok += 1
    return f"All {n_ok} sl_*.yaml configs parse with expected boolean flags."


# ----- Test runner ------------------------------------------------------------

TESTS = [
    ("FEATURE-002 AS-Norm",         test_feature_002_as_norm),
    ("FEATURE-003 per-lang eval",   test_feature_003_per_lang_eval),
    ("FEATURE-004 finetune freeze", test_feature_004_finetune_freeze),
    ("FEATURE-005 LLRD",            test_feature_005_llrd),
    ("FEATURE-006 ECAPA-TDNN",      test_feature_006_ecapa_tdnn),
    ("FEATURE-007 lang-aux",        test_feature_007_lang_aux),
    ("FEATURE-008 DANN/GRL",        test_feature_008_dann_grl),
    ("FEATURE-010 PLDA",            test_feature_010_plda),
    ("FEATURE-011 sl_dataprep",     test_feature_011_dataprep),
    ("Configs parse",               test_configs_parse),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--feature", default="", help="Substring filter")
    args = ap.parse_args()

    selected = [(n, fn) for n, fn in TESTS if args.feature.lower() in n.lower()]
    if not selected:
        print(f"No tests match --feature {args.feature!r}")
        sys.exit(1)

    print(f"Running {len(selected)} feature smoke test(s)...\n")
    n_pass = n_fail = 0
    for name, fn in selected:
        try:
            msg = fn()
            print(f"  {PASS}  {name:35s}  {msg}")
            n_pass += 1
        except Exception as e:
            print(f"  {FAIL}  {name:35s}  {type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
            n_fail += 1

    print(f"\n{n_pass}/{len(selected)} passed.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
