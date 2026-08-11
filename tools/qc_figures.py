#!/usr/bin/env python3
"""Figure and formula rendering for the QC reports.

Everything returns a self-contained SVG string, so the reports embed cleanly
with no external assets and stay crisp at any zoom.

Figures are drawn on an explicit white card with dark ink, deliberately, so a
single rendering reads correctly whether the surrounding page is in light or
dark mode. The page chrome around them is theme-aware; the plots are not.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#17222D"
INK2 = "#3A4753"
MUTED = "#566472"
SIGNAL = "#00707A"
SIGNAL_L = "#4FB3BC"
CAUTION = "#B0761A"
REJECT = "#90384A"
GRID = "#E7EBEE"
CARD = "#FFFFFF"

SERIES = ["#00707A", "#B0761A", "#90384A", "#3A6EA5", "#5C7A3F", "#7A5195"]

plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "DejaVu Sans",
    # Match the reports' body typeface (DejaVu Serif) so equations read as part
    # of the running text rather than as pasted-in images. The default
    # "dejavusans" set is what made the maths look foreign in the PDFs.
    "mathtext.fontset": "dejavuserif",
    "mathtext.default": "it",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.facecolor": CARD,
    "axes.facecolor": CARD,
    "savefig.facecolor": CARD,
    "legend.frameon": False,
})


# When True, the figure builders return the live matplotlib Figure instead of an
# SVG string, so a caller can save PNG and SVG from the same object. Needed
# because cairosvg is not installed and SVG->PNG conversion is unavailable.
RETURN_FIG = False


def _svg(fig, tight=True):
    if tight:
        fig.tight_layout(pad=0.6)
    if RETURN_FIG:
        return fig
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def _clean(ax, top_right=True):
    ax.spines["top"].set_visible(not top_right)
    ax.spines["right"].set_visible(not top_right)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8)


def formula(tex, fontsize=13, color=INK):
    """Render a LaTeX-ish expression via matplotlib mathtext (no LaTeX needed)."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color=color)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.04,
                transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


# --------------------------------------------------------------------------- #
def duration_hist(edges, counts, title="Utterance duration"):
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    widths = [(edges[i + 1] - edges[i]) * 0.92 for i in range(len(counts))]
    ax.bar(centers, counts, width=widths, color=SIGNAL, alpha=.9)
    ax.axvline(2, color=REJECT, lw=1.2, ls="--")
    ax.text(2.15, max(counts) * .92, "2 s", color=REJECT, fontsize=7.5)
    ax.set_xlabel("seconds"); ax.set_ylabel("utterances")
    ax.set_title(title); _clean(ax)
    return _svg(fig)


def lorenz(counts, gini_val, title="Utterances per speaker (Lorenz)"):
    x = np.sort(np.asarray(counts, float))
    cum = np.cumsum(x) / x.sum()
    p = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.plot(np.concatenate([[0], p]), np.concatenate([[0], cum]),
            color=SIGNAL, lw=2)
    ax.fill_between(np.concatenate([[0], p]), np.concatenate([[0], cum]),
                    np.concatenate([[0], p]), color=SIGNAL, alpha=.13)
    ax.set_xlabel("speakers (cumulative)"); ax.set_ylabel("utterances (cumulative)")
    ax.set_title(f"{title}\nGini = {gini_val:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)
    return _svg(fig)


def cosine_distributions(within, between, dprime_val, thr=None):
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    bins = np.linspace(-0.6, 1.0, 90)
    ax.hist(between, bins=bins, color=REJECT, alpha=.55, density=True,
            label=f"between-speaker  (n={len(between):,})")
    ax.hist(within, bins=bins, color=SIGNAL, alpha=.65, density=True,
            label=f"within-speaker  (n={len(within):,})")
    if thr is not None:
        ax.axvline(thr, color=INK, lw=1, ls=":")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("density")
    ax.set_title(f"Speaker separability   d' = {dprime_val:.2f}")
    ax.legend(fontsize=7.5, loc="upper left"); _clean(ax)
    return _svg(fig)


def loo_hist(loo, frac_below):
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.hist(loo, bins=80, color=SIGNAL, alpha=.85)
    ax.axvline(0.3, color=REJECT, lw=1.3, ls="--")
    ax.text(0.31, ax.get_ylim()[1] * .8,
            f"0.3\n{frac_below*100:.2f}% below", color=REJECT, fontsize=7.5)
    ax.set_xlabel("cosine to leave-one-out own-speaker centroid")
    ax.set_ylabel("utterances")
    ax.set_title("Mislabel screen: low = does not match its own speaker")
    _clean(ax)
    return _svg(fig)


def scatter_projection(xy, labels, title, subtitle="", flag=None, size=9):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    uniq = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    for i, u in enumerate(uniq):
        m = labels == u
        ax.scatter(xy[m, 0], xy[m, 1], s=size, color=cmap(i % 20),
                   alpha=.85, linewidths=.25, edgecolors="white")
    if flag is not None and flag.any():
        ax.scatter(xy[flag, 0], xy[flag, 1], s=size * 7, facecolors="none",
                   edgecolors=REJECT, linewidths=1.1, label="flagged")
        ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    return _svg(fig)


def anova_bars(conf, title="Signal metrics as speaker identity"):
    keys = [k for k in ("snr_db", "rms_dbfs", "bandwidth_hz", "crest_db",
                        "speech_ratio") if k in conf]
    vals = [conf[k]["eta_squared"] for k in keys]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    cols = [REJECT if v > .5 else (CAUTION if v > .25 else SIGNAL) for v in vals]
    ax.barh(range(len(keys)), vals, color=cols, alpha=.9)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=8)
    ax.set_xlim(0, 1)
    ax.axvline(.25, color=MUTED, lw=.8, ls=":")
    ax.axvline(.5, color=REJECT, lw=.8, ls="--")
    ax.set_xlabel(r"$\eta^2$  (variance in the metric explained by speaker)")
    ax.set_title(title)
    ax.invert_yaxis(); _clean(ax)
    return _svg(fig)


def bootstrap_ci(boot_mean, ci, mde_lo, mde_hi, eer, title="EER and its uncertainty"):
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    ax.errorbar([eer], [0], xerr=[[eer - ci[0]], [ci[1] - eer]], fmt="o",
                color=SIGNAL, capsize=5, markersize=7, lw=2)
    ax.axvspan(eer - mde_hi / 2, eer + mde_hi / 2, color=CAUTION, alpha=.13)
    ax.axvspan(eer - mde_lo / 2, eer + mde_lo / 2, color=REJECT, alpha=.10)
    ax.set_yticks([])
    ax.set_xlabel("EER (%)")
    ax.set_title(f"{title}\n95% CI {ci[0]:.2f}–{ci[1]:.2f}   "
                 f"MDE {mde_lo:.2f}–{mde_hi:.2f} pp")
    _clean(ax)
    ax.spines["left"].set_visible(False)
    return _svg(fig)


# --------------------------------------------------------------------------- #
# cross-dataset comparison figures
# --------------------------------------------------------------------------- #
def compare_bars(names, values, xlabel, title, thresholds=None, fmt="{:.2f}",
                 higher_is_better=True, figsize=(5.6, 2.8)):
    fig, ax = plt.subplots(figsize=figsize)
    order = np.argsort(values)[::-1] if higher_is_better else np.argsort(values)
    n = [names[i] for i in order]
    v = [values[i] for i in order]
    cols = []
    for x in v:
        if thresholds:
            good, bad = thresholds
            if (x >= good) if higher_is_better else (x <= good):
                cols.append(SIGNAL)
            elif (x >= bad) if higher_is_better else (x <= bad):
                cols.append(CAUTION)
            else:
                cols.append(REJECT)
        else:
            cols.append(SIGNAL)
    ax.barh(range(len(n)), v, color=cols, alpha=.92)
    for i, x in enumerate(v):
        ax.text(x + max(v) * .015, i, fmt.format(x), va="center", fontsize=7.5,
                color=INK2)
    ax.set_yticks(range(len(n)))
    ax.set_yticklabels(n, fontsize=8)
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.set_xlim(0, max(v) * 1.18)
    ax.invert_yaxis(); _clean(ax)
    return _svg(fig)


def scatter_speakers_hours(names, speakers, hours, mde=None):
    """Speakers vs hours, marker size encoding statistical power (1/MDE).

    The size scale is CLIPPED on purpose: a raw 1/MDE mapping makes the best
    corpus a blob that overruns the axes and hides its neighbours, which is
    exactly the failure the dataviz guidance warns about. Area is bounded to a
    legible range and the mapping is stated in the title.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    if mde:
        finite = [m for m in mde if m == m]
        lo, hi = (min(finite), max(finite)) if finite else (1, 1)
        sizes = []
        for m in mde:
            if m != m:
                sizes.append(60)
                continue
            # invert, then normalise to a bounded area range
            t = (1 / max(m, 1e-3) - 1 / hi) / max(1 / max(lo, 1e-3) - 1 / hi, 1e-9)
            sizes.append(50 + 300 * float(np.clip(t, 0, 1)))
    else:
        sizes = [120] * len(names)

    order = np.argsort(sizes)[::-1]          # draw big first so small stay visible
    for i in order:
        ax.scatter(speakers[i], hours[i], s=sizes[i], color=SERIES[i % len(SERIES)],
                   alpha=.55, edgecolors="white", linewidths=1.4, zorder=2)
    for i in order:
        ax.scatter(speakers[i], hours[i], s=14, color=SERIES[i % len(SERIES)],
                   zorder=4)

    # nudge labels apart so the crowded low-left corner stays readable
    offsets = {}
    for i in np.argsort(speakers):
        clash = [offsets[j] for j in offsets
                 if abs(np.log10(max(speakers[i], 1)) - np.log10(max(speakers[j], 1))) < .16
                 and abs(np.log10(max(hours[i], .1)) - np.log10(max(hours[j], .1))) < .20]
        dy = (min(clash) - 11) if clash else 8      # stack below every clash, not just the last
        offsets[i] = dy
        ax.annotate(names[i], (speakers[i], hours[i]), fontsize=7.5, color=INK2,
                    xytext=(9, dy), textcoords="offset points", zorder=5)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(min(speakers) * .55, max(speakers) * 2.6)
    ax.set_ylim(min(hours) * .5, max(hours) * 2.4)
    ax.set_xlabel("speakers (log)"); ax.set_ylabel("hours (log)")
    ax.set_title("Corpus scale   (marker area ∝ 1/MDE, clipped)")
    _clean(ax)
    return _svg(fig)


def duration_overlay(datasets):
    """datasets: list of (name, edges, counts)."""
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for i, (nm, edges, counts) in enumerate(datasets):
        c = np.array(counts, float)
        c = c / c.sum()
        centers = [(edges[j] + edges[j + 1]) / 2 for j in range(len(counts))]
        ax.plot(centers, c, marker="o", ms=3, lw=1.6,
                color=SERIES[i % len(SERIES)], label=nm)
    ax.axvline(2, color=REJECT, lw=1, ls="--")
    ax.set_xlabel("seconds"); ax.set_ylabel("fraction of utterances")
    ax.set_title("Duration profiles")
    ax.set_xlim(0, 30)
    ax.legend(fontsize=7); _clean(ax)
    return _svg(fig)


def mde_chart(names, eers, mdes, deltas=(0.2, 0.5)):
    """The decisive chart: is the effect we chase bigger than the noise floor?"""
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    order = np.argsort(mdes)
    n = [names[i] for i in order]
    m = [mdes[i] for i in order]
    cols = [SIGNAL if x <= deltas[0] else (CAUTION if x <= deltas[1] else REJECT)
            for x in m]
    ax.barh(range(len(n)), m, color=cols, alpha=.92)
    ax.axvline(deltas[0], color=SIGNAL, lw=1.2, ls="--")
    ax.axvline(deltas[1], color=REJECT, lw=1.2, ls="--")
    ax.text(deltas[0], -0.75, " 0.2 pp target", color=SIGNAL, fontsize=7)
    ax.text(deltas[1], -0.75, " 0.5 pp limit", color=REJECT, fontsize=7)
    for i, x in enumerate(m):
        ax.text(x * 1.02, i, f"{x:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(n))); ax.set_yticklabels(n, fontsize=8)
    ax.set_xlabel("minimum detectable EER difference (percentage points)")
    ax.set_title("Can this corpus resolve an ablation effect?")
    ax.invert_yaxis(); _clean(ax)
    return _svg(fig)


def grid_projections(items, kind="t-SNE", max_cols=3):
    """items: list of (name, xy, labels). Small multiples, wrapped into rows.

    Wrapping matters once there are more than ~4 corpora: a single row squeezes
    each panel until the cluster structure is unreadable, which defeats the
    point of showing the projections at all.
    """
    n = len(items)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    cmap = plt.get_cmap("tab20")
    for ax, (nm, xy, labels) in zip(axes, items):
        for i, u in enumerate(sorted(set(labels.tolist()))):
            m = labels == u
            ax.scatter(xy[m, 0], xy[m, 1], s=7, color=cmap(i % 20), alpha=.85,
                       linewidths=0)
        ax.set_title(nm, fontsize=9, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.suptitle(f"{kind} — same speaker subsample per corpus, cosine metric",
                 fontsize=10, color=INK, y=1.0)
    return _svg(fig)
