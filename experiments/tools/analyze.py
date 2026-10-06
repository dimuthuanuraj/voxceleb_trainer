#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate every experiment and run the statistical comparisons.

    python experiments/tools/analyze.py                    # full report
    python experiments/tools/analyze.py --stage A          # one stage
    python experiments/tools/analyze.py --no-figures

Outputs, under ``experiments/analysis/``::

    results_table.csv        one row per (experiment, evaluation set)
    comparisons.json         every paired-bootstrap contrast, with CIs
    report.md                the written analysis
    figures/*.png            learning curves, efficiency frontier, DET, deltas

THE STATISTICS
==============

Absolute EER is a weak instrument here.  Trials that share a speaker are
correlated, so the effective sample size is bounded by the *speaker* count:

    n_eff = m*S / (1 + (m-1) * rho)  ->  S / rho   as trials-per-speaker m grows

At S = 91 held-out Sinhala speakers and rho = 0.7 that ceiling is n_eff <= 130,
which puts the resolution of a single absolute EER at roughly 6 percentage
points -- far coarser than the differences between architectures.

The comparison that *is* well-powered is the **paired** one.  Two systems are
scored on identical trials, so writing each system's per-speaker error rate as

    e_A(s) = mu(s) + a(s),     e_B(s) = mu(s) + b(s)

the speaker-difficulty term mu(s) is common to both and cancels in the
difference.  The residual variance comes only from where the two systems
disagree, which is typically an order of magnitude smaller.  ``paired_bootstrap``
resamples *speakers* (not trials) with replacement, recomputes both systems'
EER on each resample, and reports the distribution of the difference.  A
contrast whose 95% interval excludes zero is a real ordering; one that straddles
zero is not, no matter how far apart the point estimates look.

Beyond EER, three diagnostics that say *why* a system behaves as it does:

  d'        (mu_tar - mu_non) / sqrt((var_tar + var_non)/2)
            separation of the two score distributions in units of their pooled
            spread. EER is a function of d' only when both are Gaussian, so
            comparing them exposes non-Gaussian score behaviour.

  minCllr   the application-independent information cost of the scores, after
            an optimal monotonic recalibration found by the PAV algorithm.
            EER sees only the ranking at one operating point; minCllr sees the
            whole ranking. A system with better EER but worse minCllr is
            winning at one threshold and losing overall.

  Cllr-minCllr   the calibration loss: how much is thrown away by the scores
            being mis-scaled rather than mis-ranked. This is the quantity
            AS-Norm and per-language thresholds act on.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import sys

import numpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "experiments", "analysis")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def eer_from_scores(scores, labels):
    """Equal error rate (%) by sweeping the ROC."""
    scores = numpy.asarray(scores, dtype=numpy.float64)
    labels = numpy.asarray(labels, dtype=numpy.int32)
    order = numpy.argsort(-scores)
    lab = labels[order]
    n_tar = int((lab == 1).sum())
    n_non = int((lab == 0).sum())
    if n_tar == 0 or n_non == 0:
        return float("nan")
    tp = numpy.cumsum(lab == 1)
    fp = numpy.cumsum(lab == 0)
    fnr = 1.0 - tp / n_tar   # misses at each threshold
    fpr = fp / n_non         # false alarms
    idx = int(numpy.nanargmin(numpy.abs(fnr - fpr)))
    return float(100.0 * 0.5 * (fnr[idx] + fpr[idx]))


def dprime(scores, labels):
    s = numpy.asarray(scores, dtype=numpy.float64)
    l = numpy.asarray(labels)
    t, n = s[l == 1], s[l == 0]
    if len(t) < 2 or len(n) < 2:
        return None
    pooled = numpy.sqrt((t.var(ddof=1) + n.var(ddof=1)) / 2.0)
    return float((t.mean() - n.mean()) / pooled) if pooled > 0 else None


def _pav(y, w):
    """Pool-adjacent-violators: the isotonic fit used for minCllr."""
    y = list(map(float, y))
    w = list(map(float, w))
    blocks = [[y[i], w[i]] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v0, w0 = blocks[i]
            v1, w1 = blocks[i + 1]
            merged = [(v0 * w0 + v1 * w1) / (w0 + w1), w0 + w1]
            blocks[i : i + 2] = [merged]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for v, ww in blocks:
        out.extend([v] * int(round(ww)))
    return numpy.array(out)


def cllr_pair(scores, labels):
    """Return (Cllr, minCllr) in bits.

        Cllr = 1/2 [ mean_tar log2(1 + 1/LR) + mean_non log2(1 + LR) ]

    Cllr treats the raw score as a log-likelihood ratio directly, so it charges
    for both mis-ranking and mis-scaling.  minCllr applies the optimal monotonic
    transform first (PAV), leaving only the mis-ranking part.  The gap between
    them is the calibration loss.
    """
    s = numpy.asarray(scores, dtype=numpy.float64)
    l = numpy.asarray(labels).astype(int)
    if (l == 1).sum() == 0 or (l == 0).sum() == 0:
        return None, None

    tar, non = s[l == 1], s[l == 0]
    cllr = 0.5 * (
        numpy.mean(numpy.log2(1.0 + numpy.exp(-tar)))
        + numpy.mean(numpy.log2(1.0 + numpy.exp(non)))
    )

    order = numpy.argsort(s)
    lab = l[order]
    post = _pav(lab, numpy.ones(len(lab)))
    eps = 1e-6
    post = numpy.clip(post, eps, 1 - eps)
    prior = (l == 1).mean()
    # posterior -> LLR, removing the empirical prior
    llr = numpy.log(post / (1 - post)) - numpy.log(prior / (1 - prior))
    t_llr, n_llr = llr[lab == 1], llr[lab == 0]
    min_cllr = 0.5 * (
        numpy.mean(numpy.log2(1.0 + numpy.exp(-t_llr)))
        + numpy.mean(numpy.log2(1.0 + numpy.exp(n_llr)))
    )
    return float(cllr), float(min_cllr)


def paired_bootstrap(scores_a, scores_b, labels, cluster, n_boot=2000, seed=42,
                     alpha=0.05):
    """Speaker-clustered paired bootstrap on the EER difference (A - B).

    Resampling unit is the *speaker cluster*, not the trial: trials sharing a
    speaker are not independent, and resampling them individually would
    understate the variance by roughly the design effect (a factor of ~m*rho).
    """
    scores_a = numpy.asarray(scores_a, dtype=numpy.float64)
    scores_b = numpy.asarray(scores_b, dtype=numpy.float64)
    labels = numpy.asarray(labels, dtype=numpy.int32)
    cluster = numpy.asarray(cluster)

    uniq = numpy.unique(cluster)
    index_of = {c: numpy.where(cluster == c)[0] for c in uniq}
    rng = numpy.random.default_rng(seed)

    observed = eer_from_scores(scores_a, labels) - eer_from_scores(scores_b, labels)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = numpy.concatenate([index_of[c] for c in pick])
        lab = labels[idx]
        if (lab == 1).sum() < 2 or (lab == 0).sum() < 2:
            continue
        deltas.append(eer_from_scores(scores_a[idx], lab) - eer_from_scores(scores_b[idx], lab))
    if not deltas:
        return {"error": "no usable bootstrap resamples"}
    deltas = numpy.array(deltas)
    lo, hi = numpy.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p: how often the resampled difference falls on the
    # opposite side of zero from the observed one.
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta_eer_pp": round(float(observed), 4),
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
        "p_value": round(float(min(1.0, p)), 4),
        "significant": bool(lo > 0 or hi < 0),
        "n_bootstrap": int(len(deltas)),
        "n_clusters": int(len(uniq)),
    }


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_experiments(stage=None, include_reduced=False):
    """Collect manifest + final + test_eval for every completed experiment.

    Reduced-scale runs are excluded by default: a 2-epoch smoke result sitting
    in the same table as a 60-epoch run is the kind of thing that eventually
    gets quoted by mistake.
    """
    out = {}
    for man_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "manifest.json"))):
        d = os.path.dirname(man_path)
        exp_id = os.path.basename(d)
        if not include_reduced and (exp_id.endswith("__smoke") or exp_id.endswith("__dev")):
            continue
        # ".superseded-<stamp>" dirs are the partial attempt that a re-run moved
        # aside. They hold a truncated epoch history and no final result, so
        # they must never appear in a results table beside a completed run.
        if ".superseded-" in exp_id:
            continue
        if stage and not exp_id.startswith(f"{stage}_"):
            continue
        rec = {"exp_id": exp_id, "dir": d}
        for name, key in (("manifest.json", "manifest"), ("final.json", "final"),
                          ("test_eval.json", "test_eval")):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as fh:
                        rec[key] = json.load(fh)
                except Exception:
                    pass
        if "manifest" in rec:
            out[exp_id] = rec
    return out


def epochs_of(rec):
    p = os.path.join(rec["dir"], "epochs.jsonl")
    rows = []
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def load_scores(rec, set_name):
    p = os.path.join(rec["dir"], "scores", f"{set_name}.npz")
    if not os.path.isfile(p):
        return None
    z = numpy.load(p, allow_pickle=False)
    return {
        "scores": z["scores"],
        "scores_asnorm": z["scores_asnorm"] if z["scores_asnorm"].size else None,
        "labels": z["labels"],
        "spk_a": z["spk_a"],
        "spk_b": z["spk_b"],
    }


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def build_rows(exps):
    rows = []
    for exp_id, rec in sorted(exps.items()):
        m = rec["manifest"]
        e = m["experiment"]
        card = m.get("model_card", {})
        fin = rec.get("final", {})
        best = (fin or {}).get("best") or {}
        base = {
            "exp_id": exp_id,
            "stage": e.get("stage"),
            "architecture": e.get("architecture"),
            "model": e.get("model"),
            "loss": e.get("loss"),
            "condition": e.get("condition"),
            "n_classes": e.get("n_classes"),
            "seed": e.get("seed"),
            "params_total": card.get("parameters_total"),
            "gflops_2s": card.get("gflops_per_2s_utterance"),
            "embedding_dim": card.get("embedding_dim"),
            "best_epoch": best.get("epoch"),
            "val_eer": best.get("val_eer"),
            "val_mindcf": best.get("val_mindcf"),
            "epochs_completed": fin.get("n_epochs_completed"),
            "mean_epoch_s": fin.get("mean_epoch_duration_s"),
            "wall_h": fin.get("total_wall_h"),
            "exit_code": fin.get("exit_code"),
        }
        te = rec.get("test_eval")
        if not te:
            rows.append(dict(base, eval_set=None))
            continue
        for s in te.get("sets", []):
            if "cosine" not in s:
                continue
            r = dict(base)
            r["eval_set"] = s["set"]
            r["test_eer"] = s["cosine"]["eer"]
            r["test_mindcf"] = s["cosine"]["mindcf"]
            r["test_n_trials"] = s["cosine"]["n_trials"]
            if "as_norm" in s:
                r["test_eer_asnorm"] = s["as_norm"]["eer"]
                r["test_mindcf_asnorm"] = s["as_norm"]["mindcf"]
            rows.append(r)
    return rows


def enrich_with_score_stats(exps, rows):
    """Add d', Cllr and minCllr, which need the raw score vectors."""
    cache = {}
    for r in rows:
        if not r.get("eval_set"):
            continue
        key = (r["exp_id"], r["eval_set"])
        if key not in cache:
            cache[key] = load_scores(exps[r["exp_id"]], r["eval_set"])
        z = cache[key]
        if not z:
            continue
        r["dprime"] = dprime(z["scores"], z["labels"])
        cllr, mincllr = cllr_pair(z["scores"], z["labels"])
        r["cllr"] = round(cllr, 4) if cllr is not None else None
        r["min_cllr"] = round(mincllr, 4) if mincllr is not None else None
        if cllr is not None and mincllr is not None:
            r["calibration_loss"] = round(cllr - mincllr, 4)
    return rows


def run_comparisons(exps, eval_set="test", n_boot=2000):
    """Paired bootstrap over every pair of systems sharing an evaluation set."""
    by_set = {}
    for exp_id, rec in exps.items():
        z = load_scores(rec, eval_set)
        if z is None:
            continue
        by_set.setdefault(eval_set, {})[exp_id] = z

    comparisons = []
    systems = by_set.get(eval_set, {})
    for a, b in itertools.combinations(sorted(systems), 2):
        za, zb = systems[a], systems[b]
        if len(za["labels"]) != len(zb["labels"]):
            continue  # different trial lists: not a paired comparison
        if not numpy.array_equal(za["labels"], zb["labels"]):
            continue
        res = paired_bootstrap(za["scores"], zb["scores"], za["labels"],
                               za["spk_a"], n_boot=n_boot)
        res.update({"system_a": a, "system_b": b, "eval_set": eval_set,
                    "eer_a": round(eer_from_scores(za["scores"], za["labels"]), 4),
                    "eer_b": round(eer_from_scores(zb["scores"], zb["labels"]), 4)})
        comparisons.append(res)
    return comparisons


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(exps, rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] skipped: {exc!r}")
        return []
    os.makedirs(out_dir, exist_ok=True)
    made = []

    # 1. learning curves, one panel per condition
    conds = sorted({r["condition"] for r in rows if r.get("condition")})
    if conds:
        fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 4.5),
                                 squeeze=False)
        for ax, cond in zip(axes[0], conds):
            for exp_id, rec in sorted(exps.items()):
                if rec["manifest"]["experiment"].get("condition") != cond:
                    continue
                ep = epochs_of(rec)
                xs = [e["epoch"] for e in ep if e.get("val_eer") is not None]
                ys = [e["val_eer"] for e in ep if e.get("val_eer") is not None]
                if xs:
                    ax.plot(xs, ys, marker="o", ms=2.5, lw=1.2,
                            label=rec["manifest"]["experiment"]["architecture"])
            ax.set_title(f"validation EER — {cond}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("EER (%)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(out_dir, "learning_curves.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    # 2. efficiency frontier: accuracy against compute
    pts = [r for r in rows if r.get("eval_set") == "test" and r.get("test_eer")
           and r.get("params_total")]
    if pts:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cond, marker in (("si", "o"), ("ta", "s"), ("combined", "^")):
            sub = [r for r in pts if r["condition"] == cond]
            if not sub:
                continue
            ax.scatter([r["params_total"] / 1e6 for r in sub],
                       [r["test_eer"] for r in sub], marker=marker, s=60, label=cond)
            for r in sub:
                ax.annotate(r["architecture"], (r["params_total"] / 1e6, r["test_eer"]),
                            fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("parameters (M, log scale)")
        ax.set_ylabel("held-out test EER (%)")
        ax.set_title("accuracy vs model size")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, "efficiency_frontier.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    # 3. DET curves on the shared test set
    fig, ax = plt.subplots(figsize=(6, 6))
    any_det = False
    for exp_id, rec in sorted(exps.items()):
        z = load_scores(rec, "test")
        if z is None:
            continue
        s = numpy.asarray(z["scores"], dtype=numpy.float64)
        l = numpy.asarray(z["labels"])
        order = numpy.argsort(-s)
        lab = l[order]
        n_t, n_n = (lab == 1).sum(), (lab == 0).sum()
        if not n_t or not n_n:
            continue
        fnr = 1.0 - numpy.cumsum(lab == 1) / n_t
        fpr = numpy.cumsum(lab == 0) / n_n
        ax.plot(100 * fpr, 100 * fnr, lw=1.2,
                label=f"{rec['manifest']['experiment']['architecture']}"
                      f"/{rec['manifest']['experiment']['condition']}")
        any_det = True
    if any_det:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("false alarm rate (%)")
        ax.set_ylabel("miss rate (%)")
        ax.set_title("DET — held-out test")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(out_dir, "det_curves.png")
        fig.savefig(p, dpi=150)
        made.append(p)
    plt.close(fig)
    return made


# --------------------------------------------------------------------------
def md_table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                v = f"{v:.3f}"
            cells.append("—" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(rows, comparisons, figures, path):
    exps_done = {r["exp_id"] for r in rows}
    have_test = [r for r in rows if r.get("eval_set") == "test" and r.get("test_eer")]

    lines = [
        "# Sinhala / Tamil speaker verification — experimental analysis",
        "",
        f"{len(exps_done)} experiments; {len(have_test)} with held-out test evaluation.",
        "",
        "All splits are speaker-disjoint (`experiments/tools/build_splits.py`).",
        "Model selection used validation trials only; the test set was scored once,",
        "at the validation-selected checkpoint.",
        "",
        "## 1. Headline results (held-out test set)",
        "",
    ]
    if have_test:
        # Validation EER is kept beside the test EER on purpose. It is the
        # quantity every epoch was selected on, it exists for runs that have not
        # been evaluated yet, and the val->test gap is itself informative: a
        # large gap means the validation speakers were easier than the test
        # speakers, or that selection overfitted the validation trials.
        for r in have_test:
            if r.get("val_eer") is not None and r.get("test_eer") is not None:
                r["val_to_test"] = round(r["test_eer"] - r["val_eer"], 3)
        lines.append(md_table(
            sorted(have_test, key=lambda r: (r["condition"], r.get("test_eer") or 99)),
            ["architecture", "loss", "condition", "val_eer", "test_eer",
             "val_to_test", "test_eer_asnorm", "test_mindcf", "dprime",
             "min_cllr", "params_total", "gflops_2s"],
            ["arch", "loss", "cond", "val EER %", "test EER %", "Δ val→test",
             "test EER AS-Norm %", "minDCF", "d'", "minCllr", "params",
             "GFLOPs/2s"],
        ))
    else:
        lines.append("_No test evaluations yet — run `experiments/tools/evaluate.py --all`._")

    lines += ["", "## 2. Validation results (all runs)", ""]
    val = [r for r in rows if r.get("val_eer") is not None]
    seen, uniq_val = set(), []
    for r in val:
        if r["exp_id"] not in seen:
            seen.add(r["exp_id"])
            uniq_val.append(r)
    if uniq_val:
        lines.append(md_table(
            sorted(uniq_val, key=lambda r: (r["condition"], r["val_eer"])),
            ["exp_id", "architecture", "loss", "condition", "val_eer", "val_mindcf",
             "best_epoch", "epochs_completed", "mean_epoch_s"],
            ["experiment", "arch", "loss", "cond", "val EER %", "minDCF",
             "best ep", "epochs", "s/epoch"],
        ))

    lines += ["", "## 3. Paired significance tests", "",
              "Speaker-clustered paired bootstrap on the EER difference over identical",
              "trials. `significant` means the 95% interval excludes zero.", ""]
    if comparisons:
        sig = sorted(comparisons, key=lambda c: abs(c.get("delta_eer_pp") or 0), reverse=True)
        lines.append(md_table(
            sig,
            ["system_a", "system_b", "eer_a", "eer_b", "delta_eer_pp",
             "ci95_low", "ci95_high", "p_value", "significant"],
            ["system A", "system B", "EER A", "EER B", "Δ pp", "CI low", "CI high",
             "p", "sig"],
        ))
    else:
        lines.append("_No paired comparisons available yet._")

    if figures:
        lines += ["", "## 4. Figures", ""]
        for f in figures:
            rel = os.path.relpath(f, os.path.dirname(path))
            lines.append(f"![{os.path.basename(f)}]({rel})")

    lines += [
        "",
        "## 5. How to read these numbers",
        "",
        "* **EER alone does not rank systems.** With 91 held-out Sinhala and 131 Tamil",
        "  speakers, a single absolute EER carries roughly ±5 pp of uncertainty. Use",
        "  the paired contrasts in §3; they cancel speaker difficulty and resolve",
        "  much finer differences.",
        "* **d' vs EER.** If a system has the better EER but the worse d', its score",
        "  distribution is non-Gaussian and its advantage is confined to one",
        "  operating point.",
        "* **minCllr vs EER.** minCllr integrates over all operating points. A system",
        "  ahead on EER but behind on minCllr is not the better embedding extractor.",
        "* **calibration_loss = Cllr − minCllr** is what score normalisation and",
        "  per-language thresholds can recover without touching the model.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def batch_split_diagnostic(exps, eval_set="test"):
    """Same-batch vs cross-batch impostor EER, wherever speaker ids carry a batch.

    slr127_tamil pools three collection sites (ISTL / MILE / MICI) and its
    speaker ids are '<SITE>_<number>'. An impostor pair drawn from two different
    sites can be rejected on recording channel rather than on speaker identity,
    which inflates EER downward. This reports both halves so the channel
    contribution is visible instead of being folded into one headline number.

    Returns [] for corpora whose ids have no batch prefix (slr52_sinhala,
    slceleb2026), which is itself the point: they have no such shortcut.
    """
    out = []
    for exp_id, rec in sorted(exps.items()):
        z = load_scores(rec, eval_set)
        if z is None:
            continue
        pa = numpy.array([str(x).split("_")[0] for x in z["spk_a"]])
        pb = numpy.array([str(x).split("_")[0] for x in z["spk_b"]])
        batches = set(pa.tolist()) | set(pb.tolist())
        # A prefix scheme only exists if ids actually contain '_' and there is
        # more than one distinct value; otherwise every id is its own "batch".
        if len(batches) < 2 or not any("_" in str(x) for x in z["spk_a"][:50]):
            continue
        s, l = z["scores"], z["labels"]
        same = pa == pb
        rows = {}
        for name, mask in (("all", l == 0),
                           ("same_batch", (l == 0) & same),
                           ("cross_batch", (l == 0) & ~same)):
            if mask.sum() < 50:
                continue
            sel = (l == 1) | mask
            rows[name] = {
                "n_impostors": int(mask.sum()),
                "mean_impostor_score": round(float(s[mask].mean()), 4),
                "eer": round(eer_from_scores(s[sel], l[sel]), 4),
            }
        if "same_batch" in rows and "cross_batch" in rows:
            rows["channel_inflation_factor"] = round(
                rows["same_batch"]["eer"] / max(rows["cross_batch"]["eer"], 1e-9), 2)
            out.append({"exp_id": exp_id, "eval_set": eval_set,
                        "batches": sorted(batches), **rows})
    return out


def _print_summary(rows, comparisons):
    """Compact terminal view, so results are visible without opening a file."""
    seen, uniq = set(), []
    for r in rows:
        if r["exp_id"] in seen:
            continue
        seen.add(r["exp_id"])
        uniq.append(r)
    test_by_exp = {r["exp_id"]: r for r in rows if r.get("eval_set") == "test"}

    # Columns are labelled val_/test_ explicitly. Both stages compute an EER
    # and a minDCF and they are different numbers: the val pair is produced by
    # the trainer after every epoch on the validation trials (and is what
    # selected the checkpoint), the test pair is produced once by evaluate.py on
    # the held-out test trials at that selected checkpoint. An unlabelled
    # "minDCF" column cannot be read unambiguously.
    print("\n" + "=" * 112)
    print(f"{'experiment':40s} {'cond':>9s} | {'val EER':>8s} {'val DCF':>8s} "
          f"| {'test EER':>9s} {'test DCF':>9s} {'+ASNorm':>8s} | {'ep':>4s}")
    print(f"{'':40s} {'':>9s} | {'-- every epoch --':>17s} "
          f"| {'-- once, at the best checkpoint --':>28s} | {'':>4s}")
    print("-" * 112)
    for r in sorted(uniq, key=lambda x: (x.get("condition") or "",
                                         x.get("val_eer") or 99)):
        t = test_by_exp.get(r["exp_id"], {})
        f = lambda v, w=8, p=3: (f"{v:{w}.{p}f}" if isinstance(v, (int, float))
                                 else f"{'—':>{w}}")
        print(f"{r['exp_id'][:40]:40s} {str(r.get('condition')):>9s} | "
              f"{f(r.get('val_eer'))} {f(r.get('val_mindcf'), 8, 4)} | "
              f"{f(t.get('test_eer'), 9)} {f(t.get('test_mindcf'), 9, 4)} "
              f"{f(t.get('test_eer_asnorm'))} | {str(r.get('best_epoch')):>4s}")
    print("=" * 112)
    n_eval = sum(1 for r in uniq if test_by_exp.get(r["exp_id"], {}).get("test_eer"))
    if n_eval < len(uniq):
        print(f"  {len(uniq) - n_eval} of {len(uniq)} not yet evaluated "
              f"(test columns show em-dash). Run: "
              f"python experiments/tools/evaluate.py --all --stage <X> --probes")
    if comparisons:
        sig = [c for c in comparisons if c.get("significant")]
        print(f"\npaired contrasts: {len(sig)}/{len(comparisons)} significant at 95%")
        for c in sorted(sig, key=lambda c: c["delta_eer_pp"])[:6]:
            print(f"  {c['system_a'][:34]:34s} vs {c['system_b'][:34]:34s} "
                  f"Δ{c['delta_eer_pp']:+7.3f} pp  "
                  f"[{c['ci95_low']:+.3f},{c['ci95_high']:+.3f}]")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default=None)
    ap.add_argument("--eval-set", default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default=ANALYSIS_DIR)
    ap.add_argument("--include-reduced", action="store_true",
                    help="also include __smoke / __dev runs (plumbing checks only)")
    args = ap.parse_args()

    exps = load_experiments(args.stage, include_reduced=args.include_reduced)
    if not exps:
        raise SystemExit(
            "no completed experiments found in experiments/results/ -- "
            "run a stage first (experiments/tools/run_queue.py --stage A)"
        )
    os.makedirs(args.out, exist_ok=True)

    rows = build_rows(exps)
    rows = enrich_with_score_stats(exps, rows)

    cols = sorted({k for r in rows for k in r})
    csv_path = os.path.join(args.out, "results_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    comparisons = run_comparisons(exps, args.eval_set, args.n_boot)

    batch_diag = batch_split_diagnostic(exps, args.eval_set)
    if batch_diag:
        with open(os.path.join(args.out, "batch_split_diagnostic.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(batch_diag, fh, indent=2)
        print("\nCHANNEL / COLLECTION-BATCH DIAGNOSTIC "
              "(impostors split by recording batch)")
        print(f"  {'experiment':40s} {'all':>8s} {'same':>8s} {'cross':>8s} {'infl':>6s}")
        for d in batch_diag:
            print(f"  {d['exp_id'][:40]:40s} {d['all']['eer']:>8.3f} "
                  f"{d['same_batch']['eer']:>8.3f} {d['cross_batch']['eer']:>8.3f} "
                  f"{d['channel_inflation_factor']:>5.2f}x")
        print("  same = impostors within one collection batch (channel cannot help)")
        print("  cross = impostors across batches (rejected partly on channel)")
        print("  Quote the SAME-batch column for slr127_tamil.")
    with open(os.path.join(args.out, "comparisons.json"), "w", encoding="utf-8") as fh:
        json.dump(comparisons, fh, indent=2)

    figures = [] if args.no_figures else make_figures(
        exps, rows, os.path.join(args.out, "figures")
    )
    report = os.path.join(args.out, "report.md")
    write_report(rows, comparisons, figures, report)

    _print_summary(rows, comparisons)

    print(f"experiments analysed : {len(exps)}")
    print(f"rows                 : {len(rows)}  -> {os.path.relpath(csv_path, REPO_ROOT)}")
    print(f"paired comparisons   : {len(comparisons)}")
    n_sig = sum(1 for c in comparisons if c.get("significant"))
    if comparisons:
        print(f"  significant at 95% : {n_sig}/{len(comparisons)}")
    print(f"figures              : {len(figures)}")
    print(f"report               : {os.path.relpath(report, REPO_ROOT)}")


if __name__ == "__main__":
    main()
