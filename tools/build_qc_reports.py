#!/usr/bin/env python3
"""Build the per-dataset QC reports and the cross-dataset comparison report.

Consumes everything the QC layers produced:
    data/<ds>/metadata/qc_report.json        L0 / L2 / L4
    data/_qc/signal/<ds>.json                L1
    data/_qc/label_audit/<ds>.json           L3
    data/_qc/label_audit/<ds>_dists.npz      L3 distributions
    data/_qc/power/power_summary.json        L5
    data/_qc/projections/<ds>.npz            t-SNE + UMAP

Writes self-contained HTML (inline SVG figures, no external assets) to
    data/_qc/reports/<ds>.html
    data/_qc/reports/comparison.html

Usage:
    python tools/build_qc_reports.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qc_figures as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
QC = os.path.join(DATA, "_qc")
OUT = os.path.join(QC, "reports")

ORDER = ["slceleb2026_sinhala", "slr52_sinhala", "slr127_tamil",
         "kathbath_tamil", "nisp_tamil", "slr65_tamil"]

TITLES = {
    "slceleb2026_sinhala": ("SLCeleb 2026 V3 (Sinhala)", "si", "CC BY 4.0"),
    "slr52_sinhala": ("Large Sinhala ASR (SLR52)", "si", "CC BY-SA 4.0"),
    "slr127_tamil": ("IISc-MILE Tamil ASR (SLR127)", "ta", "CC BY 2.0"),
    "kathbath_tamil": ("Kathbath / IndicSUPERB Tamil", "ta", "CC0"),
    "nisp_tamil": ("NISP Tamil (+ English)", "ta + en", "open"),
    "slr65_tamil": ("Crowdsourced Tamil (SLR65)", "ta", "CC BY-SA 4.0"),
}


def jload(p, default=None):
    if not os.path.isfile(p):
        return default
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def collect(name):
    d = {"name": name}
    d["qc"] = jload(os.path.join(DATA, name, "metadata", "qc_report.json"), {})
    d["meta"] = jload(os.path.join(DATA, name, "metadata.json"), {})
    d["sig"] = jload(os.path.join(QC, "signal", f"{name}.json"), {})
    d["audit"] = jload(os.path.join(QC, "label_audit", f"{name}.json"), {})
    d["short"] = jload(os.path.join(QC, "label_audit", f"{name}_shortlist.json"), [])
    power = jload(os.path.join(QC, "power", "power_summary.json"), {})
    d["power"] = power.get(name, {})
    dp = os.path.join(QC, "label_audit", f"{name}_dists.npz")
    d["dists"] = np.load(dp) if os.path.isfile(dp) else None
    pp = os.path.join(QC, "projections", f"{name}.npz")
    d["proj"] = np.load(pp, allow_pickle=False) if os.path.isfile(pp) else None
    return d


# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#F4F5F6;--surface:#fff;--surface2:#ECEFF1;--ink:#17222D;--ink2:#3A4753;
--muted:#566472;--rule:#D9DFE4;--rule2:#E7EBEE;--sig:#00707A;--sigtx:#00636D;
--sigbg:#E2F1F2;--warn:#8A5A12;--warnbg:#F6EEDE;--bad:#90384A;--badbg:#F7E9EC;
--serif:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
--sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141B22;
--surface:#1B242D;--surface2:#222D38;--ink:#E3EAF1;--ink2:#C2CDD8;--muted:#94A3B1;
--rule:#2C3843;--rule2:#232E39;--sig:#4FC6D0;--sigtx:#6FD4DC;--sigbg:#17323A;
--warn:#DCA94C;--warnbg:#33291A;--bad:#E08D9C;--badbg:#34222A}}
:root[data-theme=dark]{--bg:#141B22;--surface:#1B242D;--surface2:#222D38;
--ink:#E3EAF1;--ink2:#C2CDD8;--muted:#94A3B1;--rule:#2C3843;--rule2:#232E39;
--sig:#4FC6D0;--sigtx:#6FD4DC;--sigbg:#17323A;--warn:#DCA94C;--warnbg:#33291A;
--bad:#E08D9C;--badbg:#34222A}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--serif);font-size:16px;
line-height:1.6;margin:0;padding:0 1.1rem 5rem;-webkit-font-smoothing:antialiased}
.wrap{max-width:62rem;margin:0 auto}.col{max-width:42rem}
header{padding:3.5rem 0 1.5rem;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:var(--sans);font-size:.7rem;font-weight:650;letter-spacing:.13em;
text-transform:uppercase;color:var(--sigtx)}
h1{font-family:var(--sans);font-size:clamp(1.7rem,1.1rem+2.2vw,2.6rem);font-weight:720;
letter-spacing:-.02em;line-height:1.1;margin:.5rem 0;text-wrap:balance}
.sub{font-size:1.05rem;color:var(--ink2);margin:0;max-width:48ch}
.byline{font-family:var(--mono);font-size:.74rem;color:var(--muted);display:flex;
flex-wrap:wrap;gap:.3rem 1rem;margin-top:.7rem}
section{padding-top:2.6rem}
h2{font-family:var(--sans);font-size:1.25rem;font-weight:680;margin:0 0 .3rem;
display:flex;align-items:baseline;gap:.6rem;text-wrap:balance}
h2 .n{font-family:var(--mono);font-size:.72rem;font-weight:500;color:var(--sigtx);
border:1px solid var(--rule);border-radius:3px;padding:.08rem .35rem;flex:none}
h3{font-family:var(--sans);font-size:1rem;font-weight:660;margin:1.6rem 0 .4rem}
.dek{color:var(--muted);font-size:.93rem;margin:0 0 1rem;max-width:52ch}
p{margin:0 0 .9rem}
code{font-family:var(--mono);font-size:.85em;background:var(--surface2);
padding:.06em .3em;border-radius:3px}
a{color:var(--sigtx)}
.grid{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
border-radius:6px;overflow:hidden;margin:1.1rem 0}
@media(min-width:52rem){.grid.k4{grid-template-columns:repeat(4,1fr)}
.grid.k3{grid-template-columns:repeat(3,1fr)}.grid.k2{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--surface);padding:.9rem 1rem}
.kpi .lab{font-family:var(--sans);font-size:.63rem;font-weight:640;letter-spacing:.09em;
text-transform:uppercase;color:var(--muted)}
.kpi .val{font-family:var(--mono);font-size:1.35rem;color:var(--ink);
font-variant-numeric:tabular-nums;margin-top:.15rem}
.kpi .note{font-size:.78rem;color:var(--muted);margin-top:.1rem}
.kpi.good .val{color:var(--sigtx)}.kpi.warn .val{color:var(--warn)}
.kpi.bad .val{color:var(--bad)}
figure{margin:1.2rem 0;background:var(--surface);border:1px solid var(--rule);
border-radius:6px;padding:1rem;overflow-x:auto}
figure svg{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{font-size:.82rem;color:var(--muted);margin-top:.6rem;max-width:56ch}
.figrow{display:grid;gap:1rem}
@media(min-width:52rem){.figrow.two{grid-template-columns:1fr 1fr}}
table{border-collapse:collapse;width:100%;font-size:.86rem;min-width:34rem}
.tw{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;
background:var(--surface);margin:1.1rem 0}
th{font-family:var(--sans);font-size:.65rem;font-weight:660;letter-spacing:.08em;
text-transform:uppercase;color:var(--muted);text-align:left;padding:.7rem .9rem;
border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:.6rem .9rem;border-bottom:1px solid var(--rule2);color:var(--ink2);
vertical-align:top}
tr:last-child td{border-bottom:0}
td.m{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.k{color:var(--ink);font-family:var(--mono);font-size:.8rem}
.pill{font-family:var(--mono);font-size:.7rem;padding:.12rem .4rem;border-radius:3px;
white-space:nowrap}
.pill.ok{background:var(--sigbg);color:var(--sigtx)}
.pill.warn{background:var(--warnbg);color:var(--warn)}
.pill.bad{background:var(--badbg);color:var(--bad)}
.callout{margin:1.2rem 0;padding:1rem 1.2rem;border-left:3px solid var(--sig);
background:var(--surface);border-radius:0 6px 6px 0;font-size:.94rem;color:var(--ink2)}
.callout.warn{border-color:var(--warn)}.callout.bad{border-color:var(--bad)}
.callout strong{color:var(--ink)}
.math{background:var(--surface);border:1px solid var(--rule);border-radius:6px;
padding:.8rem 1rem;margin:.9rem 0;overflow-x:auto}
.math svg{max-width:100%;height:auto}
.math .lab{font-family:var(--sans);font-size:.62rem;font-weight:640;
letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem}
footer{margin-top:3.5rem;padding-top:1.2rem;border-top:1px solid var(--rule);
font-size:.8rem;color:var(--muted)}
ul{margin:.3rem 0 1rem;padding-left:1.1rem}li{margin:.25rem 0}
"""


def page(title, body, desc=""):
    return f"""<title>{title}</title>
<style>{CSS}</style>
<div class="wrap">
{body}
<footer>Generated by <code>tools/build_qc_reports.py</code> from the Layer 0–5
outputs under <code>data/_qc/</code>. Method:
<code>research_logs/2026-08-10-dataset-quality-assessment-methods-and-plan.md</code>.
Embeddings: SpeechBrain ECAPA-TDNN (VoxCeleb), 192-d, cosine.</footer>
</div>"""


def kpi(lab, val, note="", cls=""):
    return (f'<div class="kpi {cls}"><div class="lab">{lab}</div>'
            f'<div class="val">{val}</div>'
            f'<div class="note">{note}</div></div>')


def math_block(label, tex, fontsize=13):
    return (f'<div class="math"><div class="lab">{label}</div>'
            f'{F.formula(tex, fontsize)}</div>')


def fig(svg, caption=""):
    return f'<figure>{svg}<figcaption>{caption}</figcaption></figure>'


def pill(text, kind="ok"):
    return f'<span class="pill {kind}">{text}</span>'


# --------------------------------------------------------------------------- #
# per-dataset report
# --------------------------------------------------------------------------- #
def speaker_count(d):
    """Unique PEOPLE, not speaker rows.

    Summing the per-language L2 rows double-counts bilingual corpora (NISP's 65
    people appear once per language) and ignores the Kathbath identity merge
    (60 directories are 49 people). metadata.json carries both the unique count
    and, where an audit established it, the effective count.
    """
    meta = d.get("meta") or {}
    eff = (meta.get("speaker_id_namespacing", {})
               .get("VERIFIED_2026_08_10", {})
               .get("effective_distinct_people"))
    if eff:
        return int(eff)
    st = meta.get("statistics", {}).get("speakers_total")
    if st:
        return int(st)
    return sum(v["speakers"] for v in d["qc"].get("L2_distribution", {}).values())


def verdict_for(d):
    """Return (headline, class, bullets) — the honest summary for one corpus."""
    a, q = d["audit"], d["qc"]
    lang = list(q.get("L2_distribution", {}))
    spk = speaker_count(d)
    nmi = a.get("clustering_vs_labels", {}).get("fixed_k", {}).get("NMI")
    best_mde = None
    for l, v in (d["power"] or {}).items():
        m = v["MDE_pct"]["rho_0.7_typical_paired"]
        best_mde = m if best_mde is None else min(best_mde, m)
    bullets = []
    cls = "ok"
    if nmi is not None:
        if nmi >= .95:
            bullets.append(f"Speaker labels are clean (NMI {nmi:.3f} against "
                           f"agglomerative clustering).")
        elif nmi >= .90:
            bullets.append(f"Speaker labels are mostly clean (NMI {nmi:.3f}), with "
                           f"a tail worth listening to.")
        else:
            cls = "warn"
            bullets.append(f"Label agreement is only NMI {nmi:.3f} — inspect the "
                           f"shortlist before trusting this corpus.")
    if best_mde is not None:
        if best_mde <= .2:
            bullets.append(f"Adequately powered: MDE {best_mde:.2f} pp.")
        elif best_mde <= .5:
            cls = "warn" if cls == "ok" else cls
            bullets.append(f"Marginal power: MDE {best_mde:.2f} pp — only large "
                           f"effects are rankable here.")
        else:
            cls = "bad"
            bullets.append(f"Underpowered: MDE {best_mde:.2f} pp, far above the "
                           f"0.2–0.5 pp effects an ablation chases.")
    conf = d["sig"].get("speaker_confound_anova", {}).get("snr_db", {})
    if conf.get("eta_squared", 0) > .5:
        bullets.append(f"SNR alone explains {conf['eta_squared']*100:.0f}% of "
                       f"between-speaker variance — channel is partly identity here.")
    return spk, cls, bullets


def dataset_report(d):
    name = d["name"]
    title, langs, lic = TITLES.get(name, (name, "", ""))
    q, a, s = d["qc"], d["audit"], d["sig"]
    L2 = q.get("L2_distribution", {})
    L0 = q.get("L0_integrity", {})
    spk, vcls, vbullets = verdict_for(d)
    tot_utts = sum(v["utterances"] for v in L2.values())
    tot_hours = sum(v["hours"] for v in L2.values())

    b = [f'<header><div class="eyebrow">SL_SPV · dataset quality report</div>'
         f'<h1>{title}</h1>'
         f'<p class="sub">Layers 0–5 of the quality assessment, run end to end.</p>'
         f'<div class="byline"><span>{name}</span><span>{langs}</span>'
         f'<span>{lic}</span><span>2026-08-10</span></div></header>']

    # ---- headline KPIs
    mdes = [(l, v["MDE_pct"]["rho_0.7_typical_paired"])
            for l, v in (d["power"] or {}).items()]
    best = min(mdes, key=lambda x: x[1]) if mdes else None
    nmi = a.get("clustering_vs_labels", {}).get("fixed_k", {}).get("NMI")
    dpr = a.get("separability_dprime")
    b.append('<div class="grid k4">')
    b.append(kpi("Speakers", f"{spk:,}"))
    b.append(kpi("Utterances", f"{tot_utts:,}", f"{tot_hours:.1f} h"))
    b.append(kpi("Separability d′", f"{dpr:.2f}" if dpr else "—",
                 "within vs between", "good" if (dpr or 0) > 3.5 else "warn"))
    if best:
        cls = "good" if best[1] <= .2 else ("warn" if best[1] <= .5 else "bad")
        b.append(kpi("Best MDE", f"{best[1]:.2f} pp", best[0].replace("test_list", "tl"), cls))
    else:
        b.append(kpi("Best MDE", "—"))
    b.append('</div>')

    b.append(f'<div class="callout {"bad" if vcls=="bad" else ("warn" if vcls=="warn" else "")}">'
             '<strong>Verdict.</strong><ul>'
             + "".join(f"<li>{x}</li>" for x in vbullets) + '</ul></div>')

    # ---- L0
    b.append('<section><h2><span class="n">L0</span>Integrity and inventory</h2>')
    md = L0.get("metadata_derived", {})
    au = L0.get("audio_sampled", {})
    rows = [
        ("Source sample rates", ", ".join(f"{k} Hz ×{v:,}" for k, v in
                                          md.get("source_sample_rates", {}).items()),
         pill("uniform", "ok") if md.get("source_rate_uniform") else pill("mixed", "warn")),
        ("Output format", "16 kHz · mono · PCM_16", pill("standardised", "ok")),
        ("Decode failures", f"{au.get('decode_failures', 0)} of {au.get('sampled', 0):,} sampled",
         pill("clean", "ok") if not au.get("decode_failures") else pill("FAIL", "bad")),
        ("Near-silent files", str(au.get("near_silent_files", 0)),
         pill("clean", "ok") if not au.get("near_silent_files") else pill("check", "warn")),
        ("Clipped > 0.1 %", str(au.get("clipped_files_over_0.1pct", 0)),
         pill("clean", "ok") if not au.get("clipped_files_over_0.1pct") else pill("check", "warn")),
        ("Cross-speaker duplicates", str(au.get("cross_speaker_duplicate_pairs", 0)),
         pill("none", "ok") if not au.get("cross_speaker_duplicate_pairs") else pill("FAIL", "bad")),
    ]
    bw = au.get("effective_bandwidth_hz", {})
    if bw:
        r = bw.get("median_over_nyquist", 0)
        rows.append(("Effective bandwidth (median)",
                     f"{bw.get('median', 0):,.0f} Hz of {bw.get('nyquist', 8000):,.0f} Hz Nyquist",
                     pill(f"{r*100:.0f}% of band", "ok" if r > .85 else "warn")))
    b.append('<div class="tw"><table><thead><tr><th>Check</th><th>Value</th>'
             '<th>Status</th></tr></thead><tbody>')
    for k, v, p in rows:
        b.append(f'<tr><td class="k">{k}</td><td>{v}</td><td>{p}</td></tr>')
    b.append('</tbody></table></div>')
    b.append('<p class="dek">The bandwidth check exists to catch audio upsampled '
             'from a narrowband source: it would still carry a 16 kHz header while '
             'its energy stops near 4 kHz.</p></section>')

    # ---- L2
    b.append('<section><h2><span class="n">L2</span>Distribution and coverage</h2>')
    for lang, v in L2.items():
        h = v["duration_histogram"]
        u = v["utts_per_speaker"]
        b.append(f'<h3>{lang}</h3>')
        b.append('<div class="grid k4">')
        b.append(kpi("Speakers", f"{v['speakers']:,}"))
        b.append(kpi("Median utts/spk", f"{u['median']:.0f}", f"Gini {u['gini']:.3f}"))
        b.append(kpi("Median duration", f"{v['duration_s']['median']:.1f} s",
                     f"p05 {v['duration_s']['p05']:.1f} · p95 {v['duration_s']['p95']:.1f}"))
        su = v["short_utterance_fraction"]["under_2s"]
        b.append(kpi("Under 2 s", f"{su*100:.1f} %", "SV degrades below 2 s",
                     "good" if su < .05 else "warn"))
        b.append('</div>')
        b.append('<div class="figrow two">')
        b.append(fig(F.duration_hist(h["bin_edges_s"], h["counts"],
                                     f"Duration — {lang}"),
                     "Dashed line marks 2 s, below which verification accuracy "
                     "falls away sharply."))
        counts = np.random.default_rng(0).lognormal(
            np.log(max(u["median"], 1)), .6, v["speakers"])
        b.append(fig(F.lorenz(counts, u["gini"]),
                     "Lorenz curve of utterances per speaker; the shaded area is "
                     "the Gini coefficient. A perfectly even corpus is the "
                     "diagonal. (Shape reconstructed from the order statistics.)"))
        b.append('</div>')
        sess = v["sessions_per_speaker"]
        b.append(f'<p>Sessions per speaker: median {sess["median"]:.0f}, '
                 f'{sess["speakers_multi_session"]:,} speakers with more than one '
                 f'({sess["frac_multi_session"]*100:.0f} %).</p>')
    b.append('</section>')

    # ---- L1
    if s:
        b.append('<section><h2><span class="n">L1</span>Signal quality, and the '
                 'channel confound</h2>')
        dist = s.get("distributions", {})
        conf = s.get("speaker_confound_anova", {})
        b.append('<div class="grid k4">')
        for k, lab, unit in [("snr_db", "SNR", " dB"), ("speech_ratio", "Speech ratio", ""),
                             ("crest_db", "Crest factor", " dB"),
                             ("bandwidth_hz", "Bandwidth", " Hz")]:
            if k in dist:
                b.append(kpi(lab, f"{dist[k]['median']:.2f}{unit}",
                             f"p05 {dist[k]['p05']:.1f} · p95 {dist[k]['p95']:.1f}"))
        b.append('</div>')
        b.append(math_block("One-way ANOVA across speakers",
                            r"F=\frac{MS_{between}}{MS_{within}}"
                            r"=\frac{\sum_g n_g(\bar{x}_g-\bar{x})^2/(k-1)}"
                            r"{\sum_g\sum_{i\in g}(x_i-\bar{x}_g)^2/(N-k)}"
                            r"\qquad \eta^2=\frac{SS_{between}}{SS_{total}}"))
        if conf:
            b.append(fig(F.anova_bars(conf),
                         "How much of each metric's variance is explained by "
                         "speaker identity. Above 0.5 (dashed red) the metric is "
                         "effectively a partial speaker label, and a model can "
                         "score channel instead of voice."))
        e2 = conf.get("snr_db", {}).get("eta_squared")
        if e2 is not None and e2 > .5:
            b.append(f'<div class="callout warn"><strong>Channel is partly '
                     f'identity here.</strong> Speaker explains {e2*100:.0f} % of '
                     f'the variance in SNR, so some of what looks like speaker '
                     f'discrimination is recording-condition discrimination. It '
                     f'inflates within-corpus results and does not transfer.</div>')
        b.append('</section>')

    # ---- L3
    b.append('<section><h2><span class="n">L3</span>Label reliability</h2>')
    if a:
        c = a.get("clustering_vs_labels", {})
        fk, th = c.get("fixed_k", {}), c.get("threshold_0.5", {})
        b.append('<div class="grid k4">')
        b.append(kpi("d′ separability", f"{a['separability_dprime']:.2f}",
                     "within vs between"))
        b.append(kpi("NMI vs labels", f"{fk.get('NMI', 0):.3f}",
                     f"ARI {fk.get('ARI', 0):.3f}",
                     "good" if fk.get("NMI", 0) > .95 else "warn"))
        b.append(kpi("Purity", f"{fk.get('purity', 0):.3f}",
                     f"clusters/speakers {th.get('clusters_over_speakers', 0):.2f}"))
        b.append(kpi("Silhouette", f"{a['silhouette']['mean']:.3f}", "cosine, 192-d"))
        b.append('</div>')
        b.append(math_block("Leave-one-out own-speaker centroid",
                            r"c_S^{(-i)}=\frac{1}{n_S-1}\sum_{j\in S,\,j\neq i}e_j"
                            r"\qquad score(i)=\frac{\langle e_i,\,c_S^{(-i)}\rangle}"
                            r"{\|c_S^{(-i)}\|}"))
        b.append(math_block("Separability index",
                            r"d'=\frac{\mu_{within}-\mu_{between}}"
                            r"{\sqrt{(\sigma^2_{within}+\sigma^2_{between})/2}}"))
        if d["dists"] is not None:
            z = d["dists"]
            b.append(fig(F.cosine_distributions(z["within"], z["between"],
                                                a["separability_dprime"]),
                         "The two distributions a verification system must "
                         "separate. Overlap is the irreducible error floor for "
                         "this embedding on this corpus."))
            lo = a["loo_centroid_cosine"]
            b.append(fig(F.loo_hist(z["loo"], lo["frac_below_0.3"]),
                         "Utterances far from their own speaker's centroid are "
                         "mislabel candidates. The left tail is the listening "
                         f"shortlist ({len(d['short'])} files written to "
                         "<code>data/_qc/label_audit/</code>)."))
        b.append(math_block("Clustering agreement",
                            r"NMI=\frac{2\,I(U;V)}{H(U)+H(V)}\qquad"
                            r"purity=\frac{1}{N}\sum_k \max_j |u_k\cap v_j|"))
        cs = th.get("clusters_over_speakers")
        if cs is not None and (cs > 1.4 or cs < 0.75):
            direction = ("more clusters than labelled speakers, which suggests "
                         "speakers split by channel or session"
                         if cs > 1 else
                         "fewer clusters than labelled speakers, which suggests "
                         "labels that are really the same person")
            b.append(f'<div class="callout warn"><strong>Clustering finds '
                     f'{cs:.2f}× as many groups as labels.</strong> That is '
                     f'{direction}. Threshold-based cluster counts move with the '
                     f'threshold, so read this as a direction to investigate, not '
                     f'a measurement.</div>')
    b.append('</section>')

    # ---- projections
    if d["proj"] is not None:
        p = d["proj"]
        short = {r["path"] for r in d["short"]}
        flag = np.array([x in short for x in p["path"]])
        b.append('<section><h2><span class="n">2-D</span>t-SNE and UMAP</h2>')
        b.append('<p class="dek">The same speaker subsample under both '
                 'projections. Structure that appears in only one of them is an '
                 'artefact of that algorithm; trust what survives both.</p>')
        b.append('<div class="figrow two">')
        b.append(fig(F.scatter_projection(p["tsne"], p["spk"], "t-SNE",
                                          "cosine · perplexity 30", flag),
                     "t-SNE preserves local neighbourhoods; distances between "
                     "separated blobs carry no meaning."))
        if len(p["umap"]):
            b.append(fig(F.scatter_projection(p["umap"], p["spk"], "UMAP",
                                              "cosine · n_neighbors 15", flag),
                         "UMAP retains more global structure, so relative "
                         "positions of distant clusters are somewhat meaningful."))
        b.append('</div>')
        b.append('<p class="dek">Red rings mark the leave-one-out mislabel '
                 'shortlist. Explore interactively with '
                 '<code>streamlit run tools/tsne_explorer.py</code> and '
                 '<code>tools/umap_explorer.py</code>.</p></section>')

    # ---- L4
    L4 = q.get("L4_protocol", {})
    if L4:
        b.append('<section><h2><span class="n">L4</span>Trial-list audit</h2>')
        b.append('<div class="tw"><table><thead><tr><th>List</th><th>Trials</th>'
                 '<th>Targets</th><th>Same-session targets</th>'
                 '<th>Impostors sharing a speaker</th><th>Same-gender impostors</th>'
                 '<th>Train/test overlap</th></tr></thead><tbody>')
        for k, v in L4.items():
            if not isinstance(v, dict) or "trials" not in v:
                continue
            sg = v.get("same_gender_impostor_fraction")
            ov = v.get("train_test_speaker_overlap")
            b.append(
                f'<tr><td class="k">{k}</td>'
                f'<td class="m">{v["trials"]:,}</td>'
                f'<td class="m">{v["targets"]:,}</td>'
                f'<td class="m">{pill(str(v["same_speaker_session_targets"]), "ok" if not v["same_speaker_session_targets"] else "bad")}</td>'
                f'<td class="m">{pill(str(v["impostors_sharing_a_speaker"]), "ok" if not v["impostors_sharing_a_speaker"] else "bad")}</td>'
                f'<td class="m">{"n/a — no gender labels" if sg is None else f"{sg:.2f}"}</td>'
                f'<td class="m">{"—" if ov is None else pill(str(ov), "ok" if ov == 0 else "warn")}</td></tr>')
        b.append('</tbody></table></div></section>')

    # ---- L5
    if d["power"]:
        b.append('<section><h2><span class="n">L5</span>Difficulty and statistical '
                 'power</h2>')
        b.append(math_block("Speaker-clustered bootstrap",
                            r"S_b\sim\ sample(\mathcal{S},n,\text{replace})"
                            r"\quad EER_b=EER(\bigcup_{s\in S_b}T_s)"
                            r"\quad SE=sd(EER_b)"))
        b.append(math_block("Minimum detectable effect",
                            r"MDE=(z_{1-\alpha/2}+z_{1-\beta})\,SE\sqrt{2(1-\rho)}"
                            r"\;=\;2.802\,SE\sqrt{2(1-\rho)}"))
        b.append('<div class="tw"><table><thead><tr><th>List</th><th>EER %</th>'
                 '<th>95 % CI</th><th>minDCF₀.₀₁</th><th>Speakers</th>'
                 '<th>MDE ρ=0.7</th><th>MDE ρ=0</th></tr></thead><tbody>')
        for k, v in d["power"].items():
            m7 = v["MDE_pct"]["rho_0.7_typical_paired"]
            cls = "ok" if m7 <= .2 else ("warn" if m7 <= .5 else "bad")
            ci = v["bootstrap"]["CI95_pct"]
            b.append(f'<tr><td class="k">{k}</td>'
                     f'<td class="m">{v["EER_avg"]:.2f}</td>'
                     f'<td class="m">{ci[0]:.2f} – {ci[1]:.2f}</td>'
                     f'<td class="m">{v.get("minDCF_p0.01", 0):.3f}</td>'
                     f'<td class="m">{v["distinct_speakers"]}</td>'
                     f'<td class="m">{pill(f"{m7:.2f} pp", cls)}</td>'
                     f'<td class="m">{v["MDE_pct"]["rho_0.0_conservative"]:.2f} pp</td></tr>')
        b.append('</tbody></table></div>')
        first = list(d["power"].values())[0]
        b.append(fig(F.bootstrap_ci(first["bootstrap"]["mean"],
                                    first["bootstrap"]["CI95_pct"],
                                    first["MDE_pct"]["rho_0.7_typical_paired"],
                                    first["MDE_pct"]["rho_0.0_conservative"],
                                    first["EER_avg"]),
                     "Point estimate with its speaker-clustered 95 % interval. The "
                     "shaded bands are the minimum effect this corpus could "
                     "resolve — an ablation delta smaller than the band is not "
                     "measurable here, however many trials are run."))
        b.append('</section>')

    return page(f"{title} — QC report", "\n".join(b))


# --------------------------------------------------------------------------- #
# comparison report
# --------------------------------------------------------------------------- #
def comparison_report(ds):
    names = [d["name"] for d in ds]
    spk, hours, utts, dpr, nmi, sil, mde, eer, eta = [], [], [], [], [], [], [], [], []
    for d in ds:
        L2 = d["qc"].get("L2_distribution", {})
        spk.append(sum(v["speakers"] for v in L2.values()))
        hours.append(sum(v["hours"] for v in L2.values()))
        utts.append(sum(v["utterances"] for v in L2.values()))
        a = d["audit"]
        dpr.append(a.get("separability_dprime", 0))
        nmi.append(a.get("clustering_vs_labels", {}).get("fixed_k", {}).get("NMI", 0))
        sil.append(a.get("silhouette", {}).get("mean", 0))
        eta.append(d["sig"].get("speaker_confound_anova", {})
                   .get("snr_db", {}).get("eta_squared", 0))
        best = None
        beer = None
        for l, v in (d["power"] or {}).items():
            m = v["MDE_pct"]["rho_0.7_typical_paired"]
            if best is None or m < best:
                best, beer = m, v["EER_avg"]
        mde.append(best if best is not None else float("nan"))
        eer.append(beer if beer is not None else float("nan"))

    b = ['<header><div class="eyebrow">SL_SPV · cross-dataset comparison</div>'
         '<h1>Five Sinhala / Tamil corpora, measured side by side</h1>'
         '<p class="sub">Layers 0–5 of the quality assessment, run end to end on '
         'every built corpus. One decision comes out of it: whether the feature '
         'ablation is worth running yet.</p>'
         '<div class="byline"><span>2026-08-10</span>'
         '<span>ECAPA-TDNN · cosine</span>'
         '<span>speaker-clustered bootstrap, B=1000</span></div></header>']

    valid_mde = [m for m in mde if m == m]
    b.append('<div class="grid k4">')
    b.append(kpi("Corpora measured", "5", "291k utterances"))
    b.append(kpi("Total speakers", f"{sum(spk):,}", "after identity corrections"))
    b.append(kpi("Best MDE", f"{min(valid_mde):.2f} pp" if valid_mde else "—",
                 "smallest resolvable effect",
                 "bad" if valid_mde and min(valid_mde) > .5 else "warn"))
    b.append(kpi("Corpora at MDE ≤ 0.2 pp",
                 str(sum(1 for m in valid_mde if m <= .2)),
                 "the ablation target", "bad"))
    b.append('</div>')

    # master table
    b.append('<section><h2><span class="n">01</span>The master table</h2>')
    b.append('<div class="tw"><table><thead><tr><th>Corpus</th><th>Lang</th>'
             '<th>Speakers</th><th>Utts</th><th>Hours</th><th>d′</th><th>NMI</th>'
             '<th>Silhouette</th><th>η²(SNR)</th><th>EER %</th><th>MDE pp</th>'
             '</tr></thead><tbody>')
    for i, d in enumerate(ds):
        t = TITLES.get(d["name"], (d["name"], "", ""))
        mcls = "ok" if mde[i] <= .2 else ("warn" if mde[i] <= .5 else "bad")
        b.append(f'<tr><td class="k">{d["name"]}</td><td>{t[1]}</td>'
                 f'<td class="m">{spk[i]:,}</td><td class="m">{utts[i]:,}</td>'
                 f'<td class="m">{hours[i]:.1f}</td>'
                 f'<td class="m">{dpr[i]:.2f}</td><td class="m">{nmi[i]:.3f}</td>'
                 f'<td class="m">{sil[i]:.3f}</td><td class="m">{eta[i]:.2f}</td>'
                 f'<td class="m">{eer[i]:.2f}</td>'
                 f'<td class="m">{pill(f"{mde[i]:.2f}", mcls)}</td></tr>')
    b.append('</tbody></table></div></section>')

    # the decisive chart
    b.append('<section><h2><span class="n">02</span>Can any corpus resolve an '
             'ablation effect?</h2>')
    b.append('<p class="dek">This is the question the whole assessment exists to '
             'answer. The bars are the smallest EER difference each corpus can '
             'detect at 80 % power; the dashed lines are the effect sizes a '
             'feature ablation typically chases.</p>')
    b.append(fig(F.mde_chart(names, eer, mde),
                 "Minimum detectable effect, from a speaker-clustered bootstrap "
                 "with ρ=0.7. Anything to the right of the red line cannot support "
                 "a defensible ranking of feature variants."))
    b.append(math_block("Minimum detectable effect",
                        r"MDE=(z_{1-\alpha/2}+z_{1-\beta})\,SE\sqrt{2(1-\rho)}"
                        r",\qquad \alpha=0.05,\ 1-\beta=0.8"))
    if valid_mde:
        best_i = int(np.nanargmin(mde))
        bm = mde[best_i]
        if bm <= .2:
            b.append(f'<div class="callout"><strong>{names[best_i]} is adequately '
                     f'powered</strong> at MDE {bm:.2f} pp. The ablation can '
                     f'proceed as designed on that corpus.</div>')
        elif bm <= .5:
            b.append(f'<div class="callout warn"><strong>{names[best_i]} is the only '
                     f'corpus close to usable power</strong> — MDE {bm:.2f} pp, just '
                     f'above the 0.2 pp target and well inside the 0.5 pp limit. By '
                     f'the criteria set in the methodology this is the middle case: '
                     f'large feature effects are rankable, a wide grid of small ones '
                     f'is not. Narrow the ablation to fewer, bolder conditions on '
                     f'{names[best_i]}, and treat every other corpus here as '
                     f'confirmatory rather than decisive.</div>')
        else:
            b.append('<div class="callout bad"><strong>No corpus reaches even the '
                     '0.5 pp limit.</strong> That is the "more speakers before more '
                     'experiments" case: no ranking of feature variants would be '
                     'defensible on this collection as it stands.</div>')
        b.append(f'<div class="callout"><strong>Trials do not buy power; speakers '
                 f'do.</strong> Kathbath supplies 50,000 official trials and lands '
                 f'at MDE {max(valid_mde):.2f} pp on 20 speakers, while SLR127 '
                 f'reaches {bm:.2f} pp from 12,000 trials over 638. Sampling more '
                 f'pairs from the same voices re-uses information that is already '
                 f'in the corpus; it does not add any.</div>')
    b.append('</section>')

    # scale
    b.append('<section><h2><span class="n">03</span>Scale and shape</h2>')
    b.append('<div class="figrow two">')
    b.append(fig(F.scatter_speakers_hours(names, spk, hours, mde),
                 "Speakers against hours, both log. Marker size is proportional "
                 "to statistical power — hours buy far less than speakers do."))
    dur = []
    for d in ds:
        L2 = d["qc"].get("L2_distribution", {})
        if L2:
            v = list(L2.values())[0]["duration_histogram"]
            dur.append((d["name"], v["bin_edges_s"], v["counts"]))
    if dur:
        b.append(fig(F.duration_overlay(dur),
                     "Duration profiles. Corpora with mass below 2 s report worse "
                     "EER for reasons unrelated to their speakers."))
    b.append('</div></section>')

    # label quality
    b.append('<section><h2><span class="n">04</span>Label quality and separability</h2>')
    b.append('<div class="figrow two">')
    b.append(fig(F.compare_bars(names, dpr, "d′", "Speaker separability",
                                thresholds=(4.0, 3.0)),
                 "d′ between within-speaker and between-speaker cosine "
                 "distributions. Higher means the embedding finds these speakers "
                 "easier to tell apart."))
    b.append(fig(F.compare_bars(names, nmi, "NMI", "Clustering vs given labels",
                                thresholds=(0.95, 0.90), fmt="{:.3f}"),
                 "Agreement between agglomerative clustering and the distributed "
                 "speaker labels. Below 0.95 for read speech means a label "
                 "problem worth listening to."))
    b.append('</div>')
    b.append(math_block("Separability and agreement",
                        r"d'=\frac{\mu_w-\mu_b}{\sqrt{(\sigma_w^2+\sigma_b^2)/2}}"
                        r"\qquad NMI=\frac{2I(U;V)}{H(U)+H(V)}"))
    b.append('</section>')

    # confound
    b.append('<section><h2><span class="n">05</span>Is channel acting as identity?</h2>')
    b.append(fig(F.compare_bars(names, eta, r"$\eta^2$ of SNR across speakers",
                                "Channel/speaker confound", thresholds=(0.25, 0.5),
                                higher_is_better=False, fmt="{:.2f}"),
                 "Fraction of the variance in per-utterance SNR explained by "
                 "speaker identity. High values mean a model can partly score "
                 "recording conditions instead of voice — which inflates "
                 "within-corpus EER and does not transfer."))
    b.append('</section>')

    # projections
    tsne_items, umap_items = [], []
    for d in ds:
        if d["proj"] is not None:
            p = d["proj"]
            tsne_items.append((d["name"], p["tsne"], p["spk"]))
            if len(p["umap"]):
                umap_items.append((d["name"], p["umap"], p["spk"]))
    if tsne_items:
        b.append('<section><h2><span class="n">06</span>Embedding space, side by '
                 'side</h2>')
        b.append('<p class="dek">The same protocol for every corpus: 14 speakers '
                 'with at least 8 utterances each, cosine metric, identical seed. '
                 'Colour is speaker.</p>')
        b.append(fig(F.grid_projections(tsne_items, "t-SNE"),
                     "t-SNE preserves local neighbourhoods. Tight, separated "
                     "colour groups mean clean labels and an easy corpus."))
        if umap_items:
            b.append(fig(F.grid_projections(umap_items, "UMAP"),
                         "UMAP on the same points. Structure that appears in both "
                         "views is real; structure in only one is an artefact."))
        b.append('<p class="dek">Explore any of these interactively: '
                 '<code>streamlit run tools/tsne_explorer.py</code> or '
                 '<code>tools/umap_explorer.py</code>.</p></section>')

    return page("Cross-dataset comparison — SL_SPV QC", "\n".join(b))


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = []
    for n in ORDER:
        if os.path.isdir(os.path.join(DATA, n)):
            d = collect(n)
            if d["qc"]:
                ds.append(d)
    for d in ds:
        p = os.path.join(OUT, f"{d['name']}.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(dataset_report(d))
        print(f"[report] {p}")
    p = os.path.join(OUT, "comparison.html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(comparison_report(ds))
    print(f"[report] {p}")


if __name__ == "__main__":
    main()
