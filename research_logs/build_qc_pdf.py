#!/usr/bin/env python3
"""Build the dataset-quality methodology + test plan as a PDF.

Same pure-Python route as build_corpora_pdf.py: fpdf2 + matplotlib's DejaVu.
"""
import os
import re
import matplotlib
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def md(s):
    """Normalise to the markdown subset fpdf2 understands (see build_corpora_pdf)."""
    s = s.replace("`", "")
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"__\1__", s)
    return s.replace("--", "‑‑")


FONTS = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

INK, INK2, MUTED = (0x17, 0x22, 0x2D), (0x3A, 0x47, 0x53), (0x56, 0x64, 0x72)
SIGNAL, CAUTION, REJECT = (0x00, 0x70, 0x7A), (0x8A, 0x5A, 0x12), (0x90, 0x38, 0x4A)
RULE, RULESOFT = (0xD9, 0xDF, 0xE4), (0xE7, 0xEB, 0xEE)
SIGTINT = (0xE2, 0xF1, 0xF2)

M = 18
W = 210 - 2 * M


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("sans", "", 7)
        self.set_text_color(*MUTED)
        self.set_xy(M, 10)
        self.cell(W, 4, "Dataset Quality Assessment for Speaker Verification")
        self.set_draw_color(*RULESOFT)
        self.set_line_width(0.2)
        self.line(M, 15, 210 - M, 15)
        self.set_xy(M, self.t_margin)

    def footer(self):
        self.set_y(-14)
        self.set_font("mono", "", 7)
        self.set_text_color(*MUTED)
        self.cell(W, 4, f"{self.page_no()}", align="C")


pdf = Doc("P", "mm", "A4")
pdf.set_auto_page_break(True, margin=20)
pdf.set_margins(M, 22, M)
for fam, st, f in [
    ("serif", "", "DejaVuSerif.ttf"), ("serif", "B", "DejaVuSerif-Bold.ttf"),
    ("serif", "I", "DejaVuSerif-Italic.ttf"), ("serif", "BI", "DejaVuSerif-BoldItalic.ttf"),
    ("sans", "", "DejaVuSans.ttf"), ("sans", "B", "DejaVuSans-Bold.ttf"),
    ("sans", "I", "DejaVuSans-Oblique.ttf"), ("sans", "BI", "DejaVuSans-BoldOblique.ttf"),
    ("mono", "", "DejaVuSansMono.ttf"), ("mono", "B", "DejaVuSansMono-Bold.ttf"),
    ("mono", "I", "DejaVuSansMono-Oblique.ttf"), ("mono", "BI", "DejaVuSansMono-BoldOblique.ttf"),
]:
    pdf.add_font(fam, st, os.path.join(FONTS, f))

pdf.set_title("Dataset Quality Assessment for Speaker Verification")
pdf.set_author("SL_SPV")


def space(h):
    pdf.set_y(pdf.get_y() + h)


def need(h):
    if pdf.get_y() + h > 297 - 20:
        pdf.add_page()


def h2(num, txt):
    need(28)
    space(5)
    pdf.set_draw_color(*INK)
    pdf.set_line_width(0.5)
    pdf.line(M, pdf.get_y(), 210 - M, pdf.get_y())
    space(3.5)
    y = pdf.get_y()
    pdf.set_font("mono", "", 8)
    pdf.set_text_color(*SIGNAL)
    pdf.set_xy(M, y)
    pdf.cell(13, 6, num)
    pdf.set_font("sans", "B", 13)
    pdf.set_text_color(*INK)
    pdf.set_xy(M + 13, y)
    pdf.multi_cell(W - 13, 6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(1)


def h3(txt, color=None):
    need(16)
    space(2.5)
    pdf.set_font("sans", "B", 10)
    pdf.set_text_color(*(color or INK))
    pdf.set_x(M)
    pdf.multi_cell(W, 5.2, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(1)


def dek(txt):
    pdf.set_font("serif", "I", 9)
    pdf.set_text_color(*MUTED)
    pdf.set_x(M)
    pdf.multi_cell(W, 4.6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(2)


def body(txt, size=9.3, color=INK2, lh=4.8, indent=0):
    pdf.set_font("serif", "", size)
    pdf.set_text_color(*color)
    pdf.set_x(M + indent)
    pdf.multi_cell(W - indent, lh, md(txt), new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                   markdown=True)
    space(1.5)


def bullets(items, size=9, lh=4.6):
    for it in items:
        need(12)
        y = pdf.get_y()
        pdf.set_font("serif", "", size)
        pdf.set_text_color(*SIGNAL)
        pdf.set_xy(M + 1, y)
        pdf.cell(4, lh, "•")
        pdf.set_text_color(*INK2)
        pdf.set_xy(M + 5, y)
        pdf.multi_cell(W - 5, lh, md(it), new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                       markdown=True)
        space(0.8)


def rule(color=RULESOFT, w=0.2):
    pdf.set_draw_color(*color)
    pdf.set_line_width(w)
    pdf.line(M, pdf.get_y(), 210 - M, pdf.get_y())


def table(cols, header, rows, fsize=7.6, hsize=6.5):
    need(16)
    pdf.set_font("sans", "B", hsize)
    pdf.set_text_color(*MUTED)
    y = pdf.get_y()
    x = M
    for c, h in zip(cols, header):
        pdf.set_xy(x, y)
        pdf.multi_cell(c, 4, h, new_x=XPos.RIGHT, new_y=YPos.TOP)
        x += c
    pdf.set_y(y + 5)
    rule(RULE)
    space(1.4)
    for r in rows:
        # measure tallest cell
        heights = []
        pdf.set_font("serif", "", fsize)
        for c, v in zip(cols, r):
            heights.append(len(pdf.multi_cell(c - 2, 4.1, md(str(v)), dry_run=True,
                                              output="LINES", markdown=True)))
        h = max(heights) * 4.1 + 1.6
        need(h + 3)
        y = pdf.get_y()
        x = M
        for i, (c, v) in enumerate(zip(cols, r)):
            pdf.set_font("mono" if i == 0 else "serif", "B" if i == 0 else "",
                         fsize - (0.4 if i == 0 else 0))
            pdf.set_text_color(*(INK if i == 0 else INK2))
            pdf.set_xy(x, y)
            pdf.multi_cell(c - 2, 4.1, md(str(v)), new_x=XPos.RIGHT, new_y=YPos.TOP,
                           markdown=True)
            x += c
        pdf.set_y(y + h)
        rule()
    space(2)


def callout(title, txt, color=SIGNAL):
    need(20)
    space(1)
    y0 = pdf.get_y()
    pdf.set_x(M + 4)
    pdf.set_font("serif", "", 9.2)
    pdf.set_text_color(*INK2)
    pdf.multi_cell(W - 4, 4.7, md(f"**{title}** {txt}"),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT, markdown=True)
    y1 = pdf.get_y()
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.8)
    pdf.line(M, y0, M, y1)
    space(2)


# =========================== page 1 ===========================
pdf.add_page()
space(12)
pdf.set_font("sans", "B", 7)
pdf.set_text_color(*SIGNAL)
pdf.set_x(M)
pdf.cell(W, 4, " ".join("SL_SPV · BENCHMARK QUALITY".upper()),
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(3)
pdf.set_font("sans", "B", 21)
pdf.set_text_color(*INK)
pdf.set_x(M)
pdf.multi_cell(W, 9.2, "Dataset quality assessment\nfor speaker verification",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(2)
pdf.set_font("sans", "", 12)
pdf.set_text_color(*MUTED)
pdf.set_x(M)
pdf.multi_cell(W, 6, "Methods, and a structured plan to test all eight corpora",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(3.5)
pdf.set_font("serif", "", 10.5)
pdf.set_text_color(*INK2)
pdf.set_x(M)
pdf.multi_cell(W * 0.86, 5.4,
               "The companion log answered which datasets to use. This one answers "
               "whether they are any good — and whether they can support the feature "
               "ablation we actually want to run.",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(4)
rule(RULE, 0.3)
space(2.5)
pdf.set_font("mono", "", 7.5)
pdf.set_text_color(*MUTED)
pdf.set_x(M)
pdf.cell(W, 4, "2026-08-10     methodology + plan, nothing executed yet     "
               "companion: 2026-08-10-open-si-ta-sv-corpora.md",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(5)

# =========================== Part I ===========================
h2("I", "What quality means here")
body("The instinct is to reach for audio-quality metrics. For speaker verification "
     "that is the least important axis. A corpus of clean studio speech can be "
     "worthless for verification, and a corpus of noisy YouTube audio can be "
     "excellent. Four things actually decide whether an SV dataset is sound — "
     "roughly in order of how badly they bite.")

table([9, 42, 55, 68],
      ["#", "FAILURE MODE", "WHAT IT LOOKS LIKE", "WHY IT IS FATAL"],
      [["1", "Label noise",
        "Two people sharing one speaker id, or one person split across two ids",
        "Corrupts targets *and* impostors. A mislabelled impostor that is really a "
        "target puts a hard ceiling on measurable EER. Removing noisy VoxCeleb2 "
        "samples alone bought ~5.9 % relative improvement in one study."],
       ["2", "Session / channel confound",
        "Target pairs cut from the same recording",
        "The model scores channel, not speaker. 8–17 % of VoxCeleb1-H target pairs "
        "have this. EER looks great and means nothing."],
       ["3", "Insufficient power",
        "20–50 speakers, however many trials",
        "Confidence intervals wider than the effects being ranked. **This is the one "
        "that decides whether our ablation is worth running at all.**"],
       ["4", "Distribution shift / bias",
        "Trained on Indian Tamil, deployed on Sri Lankan Tamil",
        "Headline EER hides subgroup failure. Female error rates 49 % higher than "
        "male in one challenge-wide audit."]])

callout("Audio quality matters mainly as a confound detector.",
        "If corpus A is uniformly cleaner than corpus B, an embedding separates them "
        "trivially, and any cross-corpus generalisation result is measuring codec, "
        "not voice.")

body("The assessment is therefore six layers, cheapest and most decisive first. "
     "**Layers 0–2 need no GPU and no model.** Layer 3 is where the real answers are.")

# =========================== Part II ===========================
h2("II", "The methods, layer by layer")

h3("Layer 0 — Integrity and inventory")
dek("Is the data physically what the metadata claims?")
table([48, 78, 48],
      ["CHECK", "METHOD", "RED FLAG"],
      [["Decodability", "soundfile.read every file", "any failure"],
       ["Format uniformity", "sample rate, channels, bit depth histogram",
        "more than one value per corpus"],
       ["Near-silent files", "RMS below −60 dBFS over the whole file", "> 0.1 % of files"],
       ["Exact duplicates", "MD5 over *decoded* PCM, not the container",
        "any cross-speaker duplicate"],
       ["Effective bandwidth", "mean power spectrum; rolloff where energy falls "
        "50 dB below peak", "rolloff far below Nyquist → upsampled narrowband audio"]])
body("The bandwidth check is the one people skip and regret. Audio resampled up "
     "from 8 kHz telephone still carries a 16000 header; the model sees a brick wall "
     "at 4 kHz and learns *this is corpus X*. slr65_tamil was resampled 48→16 kHz "
     "here, so its ceiling is 8 kHz by construction — expected and fine. We are "
     "hunting *unexpected* band-limiting and internal inconsistency.")

h3("Layer 1 — Signal quality, non-intrusively")
dek("How clean is the audio, and is cleanliness confounded with speaker identity?")
body("No clean reference exists for any of these corpora, so every metric must be "
     "reference-free: WADA-SNR or NIST STNR; clipping rate; speech ratio via VAD; "
     "LUFS loudness; and **torchaudio.pipelines.SQUIM_OBJECTIVE / SQUIM_SUBJECTIVE**, "
     "which estimate STOI, PESQ, SI-SDR and MOS with no reference. SQUIM is chosen "
     "over DNSMOS and NISQA purely because it ships inside the already-installed "
     "torchaudio 2.8 and adds no dependency.")
callout("The critical analysis is not the mean — it is the variance within a speaker.",
        "If a speaker's utterances all share one SNR and one loudness while different "
        "speakers differ, then SNR *is* speaker id and the model can cheat. Compute "
        "the between- vs within-speaker variance ratio (a one-way ANOVA F) for each "
        "signal metric. A high F on SNR warns that target trials are partly solvable "
        "from channel alone.", CAUTION)

h3("Layer 2 — Distribution and coverage")
dek("Is the speaker population shaped so a meaningful trial list can even be drawn?")
bullets([
    "Speakers; utterances per speaker (min / median / max); **Gini coefficient** of "
    "utterances per speaker — imbalance in one number.",
    "**Duration histogram, and the fraction below 2 s and 4 s.** SV accuracy collapses "
    "on short utterances; a corpus with a 2.5 s median reports a bad EER for reasons "
    "unrelated to its speakers.",
    "Sessions per speaker, and how many speakers have ≥ 2 — this decides how many can "
    "supply cross-session targets at all.",
    "Gender balance, and gender × utterance count.",
    "Duration × speaker correlation — a confound if a few speakers own all the long files.",
])
body("All of this is computable from metadata/utterances.csv, which every dataset "
     "folder already carries. **Layer 2 costs minutes and needs no audio.**")

h3("Layer 3 — Label reliability via speaker embeddings")
dek("Are the speaker labels actually correct? This is the layer that finds real problems.")
bullets([
    "**Extract one embedding per utterance** with a strong pretrained model, ideally "
    "two architecturally different ones so findings are not model artefacts. "
    "tools/zeroshot_eval.py already does extraction with caching.",
    "**Within- vs between-speaker similarity.** Plot both cosine distributions. Heavy "
    "overlap means either hard data or broken labels — and layers 4–5 cannot tell you which.",
    "**Leave-one-out centroid outliers.** For each utterance, cosine to its own "
    "speaker's centroid computed without it; rank ascending. The bottom tail is the "
    "mislabel candidate list. **Then actually listen to the top ~30** — this step is "
    "not automatable and is the only way to turn a suspicion into a fact.",
    "**Per-speaker silhouette score.** Low silhouette means acoustically unremarkable "
    "or two people in one id.",
    "**Clustering vs labels.** Agglomerative clustering (cosine, average linkage), "
    "compared to labels via NMI, ARI, V-measure, purity. Clusters >> speakers → one "
    "person split across ids. Clusters << speakers → distinct ids collapsing, i.e. "
    "probable duplicates.",
    "**Visual check** with t-SNE or UMAP on a speaker subsample.",
])
callout("Thresholds must be set per corpus type, not globally.",
        "Read speech recorded in one sitting should cluster almost perfectly "
        "(NMI > 0.95); anything less is a real label problem. In-the-wild corpora "
        "legitimately score lower, so the same number means different things.")

h3("Layer 4 — Protocol and trial-list audit")
dek("Is the benchmark we built out of this corpus honest? A trial list can be wrong "
    "in ways no model will ever reveal.")
table([84, 90],
      ["CHECK", "REQUIREMENT"],
      [["Target / impostor counts and ratio", "matches the design"],
       ["Self-pairs (a == b)", "zero"],
       ["Duplicate unordered pairs", "zero"],
       ["Same-(speaker, session) target pairs", "zero, wherever sessions are real"],
       ["Same-gender impostor fraction", "1.0 where gender is known; documented otherwise"],
       ["Speaker overlap between train_list and test_list", "zero, or explicitly declared"],
       ["Utterance reuse frequency", "no small set of files dominating trials"],
       ["Enrol / test duration matching", "no systematic asymmetry"]])
body("The train/test overlap check becomes essential the moment we *combine* corpora "
     "— training on SLR127 + Kathbath and evaluating on a merged list is exactly where "
     "leakage sneaks in. Note a subtlety already hit in this repo: comparing session "
     "*directory names* is wrong when session names are shared across speakers "
     "(SLR127 uses ISTL/MICI/MILE). The key must be the (speaker, session) pair.")

h3("Layer 5 — Difficulty calibration and statistical power")
dek("How hard is this benchmark, and can it detect the effects we care about?")
body("**Difficulty.** Run 2–3 pretrained models zero-shot; report EER_avg, EER_max and "
     "minDCF at p_target 0.01 and 0.05 — all of which tools/zeroshot_eval.py already "
     "computes. Express results as a ratio against the same models' VoxCeleb-O numbers "
     "for a corpus-independent difficulty index. Report **Cllr and min Cllr** too; the "
     "gap between them is miscalibration, which our earlier finding on cross-language "
     "calibrator swaps (Cllr inflated up to 5.7×) makes directly relevant.")
callout("Power is the decisive analysis, and the bootstrap must be speaker-clustered.",
        "Resample *speakers* with replacement — not trials — rebuild the trial subset, "
        "recompute EER, repeat B = 1000, take the 2.5 / 97.5 percentiles. Resampling "
        "trials independently is wrong because trials from one speaker are correlated, "
        "and it understates the interval, often badly when speakers are few.")
body("From that, derive the **minimum detectable effect (MDE)**: the smallest EER "
     "difference the corpus can resolve at 80 % power, α = 0.05, for a paired "
     "comparison on identical trials. Then state plainly whether the ablation deltas we "
     "expect (~0.2–0.5 % EER) exceed it. This turns *50 speakers feels too few* into a "
     "number, and it is the single most useful output of the exercise.")

h3("Layer 6 — Bias and cross-corpus transfer")
dek("Whose voices does this work for, and does it survive a domain change?")
bullets([
    "**Subgroup EER** by gender and, where known, dialect — reporting per-subgroup "
    "FMR/FNMR at a *shared* threshold, not just per-subgroup EER, because a single "
    "global operating point is what a deployed system actually uses.",
    "**Fairness Discrepancy Rate (FDR)** for a one-number summary, following Hutiri "
    "& Ding's bias-quantification framework.",
    "**Cross-corpus matrix**: train on A, evaluate on B for every ordered pair. The "
    "Indian-Tamil-train / Sri-Lankan-Tamil-test cell is the number this project is "
    "positioned to publish.",
    "**Channel probe**: train a logistic regression on embeddings to predict "
    "corpus-of-origin. Near-perfect accuracy means the embedding encodes channel as "
    "much as identity.",
])

h3("Reporting standard")
body("Findings belong in each corpus's existing metadata.json and README.md rather "
     "than a separate report — the datasheet lives with the data. Framing follows "
     "**Datasheets for Datasets** (Gebru et al.) and **Data Statements** (Bender & "
     "Friedman). Our metadata.json already covers provenance, licence, session "
     "semantics and caveats; layers 0–6 add the *measured* half.")

# =========================== Part III ===========================
pdf.add_page()
h2("III", "Structured test plan")

h3("Phasing")
dek("Ordered so the cheapest checks can kill a corpus before expensive ones run.")
table([16, 26, 40, 26, 66],
      ["PHASE", "LAYERS", "COMPUTE", "WALL-CLOCK", "GATE"],
      [["P0", "L0 + L2", "CPU, metadata only", "~2 h",
        "integrity clean, distribution documented"],
       ["P1", "L4", "CPU, lists only", "~2 h",
        "no leakage, no same-session targets"],
       ["P2", "L1", "CPU (GPU faster)", "~1 day",
        "quality/speaker confound quantified"],
       ["P3", "L3", "**GPU**", "~2 days",
        "label noise bounded, listened-to sample"],
       ["P4", "L5", "GPU (reuses P3)", "~1 day",
        "**MDE per corpus → go/no-go for the ablation**"],
       ["P5", "L6", "GPU", "~2 days",
        "subgroup + cross-corpus gaps reported"]])
body("P0 and P1 need nothing beyond this node. P3 onward needs the GPU offload path "
     "(tools/gpurun.sh — this node has no GPU); NFS is mounted at the same path on "
     "the compute nodes, so no data moves. Embeddings extracted in P3 are cached and "
     "reused by P4 and P5 — extract once.")

h3("Per-dataset plan")
dek("Every corpus gets L0/L2/L4. The rest is targeted at each corpus's specific known "
    "risk, so effort goes where it can change a decision.")

DATASETS = [
    ("slr52_sinhala", "HIGH", "478 spk · 185k utts · 224.5 h",
     "The entire Sinhala side rests on it. Speaker ids are anonymised crowdsourcing "
     "accounts with **no upstream guarantee that one account is one person**; no gender "
     "labels; single sitting.",
     "Full L0; L2 with attention to the duration tail; **L3 in full** — this is where a "
     "clustering-vs-labels check is most valuable, precisely because nothing guarantees "
     "account = person. Expect NMI > 0.95. L1 on a 5k-utterance sample. "
     "**Red flag that would change plans:** clusters >> speakers, implying shared or "
     "reused accounts."),
    ("slr127_tamil", "HIGH", "531 spk · 89k utts · 150.1 h",
     "Largest Tamil pool and the only read-speech corpus with real sessions. But the "
     "speaker-id convention was **derived by us, not documented upstream**, and "
     "prefix-as-session is an inference.",
     "Full L0 including the **bandwidth check** (three batches may differ in equipment); "
     "**L3 with clustering split by prefix** — this directly tests the session "
     "hypothesis: if ISTL and MILE recordings of one speaker cluster apart, the prefix "
     "is a genuine channel change, validating the mapping *and* quantifying the effect. "
     "L1 **stratified by prefix** is the physical evidence. This corpus deserves the "
     "most careful L3 because our confidence rests on inference, not documentation."),
    ("kathbath_tamil", "MED-HIGH", "60 spk · 7k utts · 13.2 h",
     "The comparability anchor. Lossy AAC; only 20 speakers per eval split; 11 numeric "
     "ids recur across splits, namespaced apart on an **unverified assumption** that "
     "they are different people.",
     "Full L0/L2/L4; **L3 with one extra test — embed kbv84, kbk84, kbt84 and check "
     "whether same-numeric-id speakers across splits are in fact the same person.** "
     "High cosine similarity would mean our namespacing silently created three "
     "identities for one speaker. Concrete, falsifiable, worth answering early. L5 "
     "matters here: 20 speakers with 50k trials is the textbook case of trial count "
     "overstating power. **Do not modify the official lists** on any finding — document."),
    ("nisp_tamil", "MEDIUM", "65 spk (all bilingual) · 4.9k utts · 13.3 h",
     "Small, but the only cross-lingual probe. English is accented, not a native control.",
     "L0/L2/L4; **L3 focused on the cross-lingual question** — compare "
     "within-speaker-within-language, within-speaker-cross-language, and "
     "between-speaker similarity distributions. The gap between the first two *is* the "
     "language effect on the embedding, measured directly, with no trial list needed. "
     "L6 is unusually feasible: age, height and region ship with the corpus."),
    ("slr65_tamil", "LOW-MED", "50 spk · 4.3k utts · 7.1 h",
     "Kept for v0 continuity only.",
     "L0/L2/L4 for completeness; L3 cheap because it is small; **L5 to put a number on "
     "how underpowered v0 was.** That number retroactively frames every v0 result and "
     "belongs in the thesis. Not worth deep L1/L6 investment."),
    ("commonvoice_tamil", "MEDIUM", "not yet fetched",
     "client_id is an *account*, not verifiably a person — the highest label-noise risk "
     "in the collection. Wildly heterogeneous devices.",
     "L0 with **bandwidth check as a first-class concern**; L1 expecting high variance "
     "— and here high variance is a *feature*, since device diversity is why we want "
     "it; **L3 mandatory before any use**, with the account-vs-person question explicit. "
     "L2 should set --min-utts empirically rather than the current default of 4."),
    ("sita_sinhala_tamil", "MEDIUM", "not yet fetched",
     "For diarization and domain probing; **not** an SV benchmark.",
     "L0/L1/L2 only. **L4 and L5 do not apply** — there is no trial list, by design. "
     "One bespoke analysis is worth doing: **cross-recording identity linking.** Embed "
     "all segments, cluster across recordings, and see whether the same public figures "
     "recur. If they do, that is the raw material for the first genuine Sri Lankan SV "
     "benchmark — and the measurement tells you whether the manual effort is worth it. "
     "That is a research contribution, not a QA step."),
    ("slceleb_sinhala_tamil", "HIGHEST", "audio not on mounts",
     "The reference set, and the only corpus whose absolute EER means what a reader "
     "will assume. YouTube-derived pipelines are exactly where label noise lives — "
     "VoxCeleb's lineage proves it — and unlike VoxCeleb1 this corpus has **no "
     "published cleaning pass**.",
     "**All six layers, in full.** Nothing else carries as much weight per speaker. "
     "L3 here is not optional. The L6 cross-corpus matrix against SLR127/Kathbath is "
     "the headline experiment."),
]
for name, prio, scale, risk, plan in DATASETS:
    need(42)
    space(2)
    rule()
    space(2.2)
    y = pdf.get_y()
    pdf.set_font("sans", "B", 10)
    pdf.set_text_color(*INK)
    pdf.set_xy(M, y)
    pdf.cell(W - 34, 5, name)
    pcol = {"HIGHEST": REJECT, "HIGH": REJECT, "MED-HIGH": CAUTION,
            "MEDIUM": CAUTION, "LOW-MED": MUTED}[prio]
    pdf.set_font("sans", "B", 6.5)
    pdf.set_text_color(*pcol)
    pdf.set_xy(210 - M - 34, y + 0.6)
    pdf.cell(34, 4, prio, align="R")
    pdf.set_y(y + 5.4)
    pdf.set_font("mono", "", 7)
    pdf.set_text_color(*MUTED)
    pdf.set_x(M)
    pdf.cell(W, 4, scale, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(1)
    body("**Risk.** " + risk, size=8.8, lh=4.4)
    body("**Run.** " + plan, size=8.8, lh=4.4)

h3("Acceptance gates")
dek("To be revised once P0/P2 show what the corpora actually look like — and split by "
    "corpus type, because read speech and in-the-wild are not comparable.")
table([76, 50, 48],
      ["GATE", "READ-SPEECH", "IN-THE-WILD"],
      [["Decode failures", "0", "0"],
       ["Cross-speaker exact duplicates", "0", "0"],
       ["Utterances < 2 s", "< 5 %", "< 10 %"],
       ["Label NMI (AHC vs labels)", "> 0.95", "> 0.80"],
       ["Centroid-outlier tail flagged for listening", "bottom 0.5 %", "bottom 1 %"],
       ["Same-(speaker, session) target pairs", "0", "0"],
       ["Train/test speaker overlap", "0", "0"],
       ["Same-gender impostor fraction", "1.0 where known", "1.0 where known"]])
body("A corpus failing a gate is **not discarded** — it is documented and "
     "down-weighted, and the failure goes into its metadata.json caveats.")

h3("Deliverables")
bullets([
    "**tools/dataset_qc.py** — L0/L2/L4, pure CPU, emits metadata/qc_report.json per "
    "dataset plus a cross-corpus summary.",
    "**tools/signal_quality.py** — L1: WADA-SNR, clipping, VAD, SQUIM, sampled.",
    "**tools/label_audit.py** — L3: consumes cached embeddings from zeroshot_eval.py; "
    "emits outlier lists, clustering metrics, and a listen-to-me shortlist of paths.",
    "**tools/eer_power.py** — L5: speaker-clustered bootstrap CIs and MDE.",
    "A results log plus a `quality` block added to each corpus's metadata.json.",
])

h3("What would change the ablation plan", REJECT)
body("The point of all this is one decision. Three outcomes are actionable:")
table([54, 120],
      ["IF MDE IS…", "THEN"],
      [["below ~0.2 % EER",
        "the ablation is properly powered and proceeds as designed"],
       ["0.3 – 0.5 % EER",
        "only large feature effects are rankable — narrow the ablation to fewer, "
        "bolder conditions rather than a wide grid"],
       ["above 0.5 % EER",
        "no amount of clever features produces a defensible ranking; the honest move "
        "is **more speakers before more experiments**"]])
callout("Layer 3 can also invalidate a corpus outright.",
        "Better to find that in P3 than in peer review.", REJECT)

# =========================== Part IV / V ===========================
h2("IV", "Environment notes")
dek("Verified on this host, 2026-08-10.")
bullets([
    "torch 2.8.0+cu128, torchaudio 2.8.0 — **SQUIM_OBJECTIVE and SQUIM_SUBJECTIVE "
    "import cleanly**.",
    "sklearn 1.5.1 — TSNE, AgglomerativeClustering, silhouette, NMI/ARI all present.",
    "speechbrain present; wespeaker, pyannote.audio and umap-learn absent (all optional).",
    "**No GPU on this node** (torch.cuda.is_available() is False). L3–L6 go through "
    "tools/gpurun.sh; NFS is mounted at the same path on compute nodes, so no data moves.",
    "Reusable already: tools/zeroshot_eval.py (cached embedding extraction, EER/minDCF), "
    "tools/check_wav_integrity.py, tools/sl_dataprep.py.",
])

h2("V", "References")
REFS = [
    ("Hutiri, Ding (2022)", "Bias in Automated Speaker Recognition. FAccT 2022. arXiv:2201.09486"),
    ("Hutiri et al. (2022)", "Interspeech — same-recording confound in VoxCeleb trial pairs"),
    ("Kumar et al. (2023)", "TorchAudio-Squim: Reference-less Speech Quality and Intelligibility. arXiv:2304.01448"),
    ("Gebru et al. (2021)", "Datasheets for Datasets. CACM 64(12)"),
    ("Bender, Friedman (2018)", "Data Statements for NLP. TACL 6"),
    ("Tong et al. (2022)", "Inconsistency Ranking-based Noisy Label Detection. arXiv:2212.00239"),
    ("CEC (2024)", "A Noisy Label Detection Method for Speaker Recognition. arXiv:2406.13268"),
    ("Fathan et al. (2025)", "Automatic Labeling and Correction of Noisy Labels for Robust SSL Speaker Verification. Interspeech 2025"),
    ("Bisani, Ney (2004)", "Bootstrap Estimates for Confidence Intervals in ASR Performance Evaluation. ICASSP 2004"),
    ("Chung et al. (2019)", "VoxSRC 2019. arXiv:1912.02522 — dev/test overlap across VoxCeleb1/2/SITW"),
    ("NIST SRE", "evaluation plans — reference for trial protocol and DCF reporting"),
    ("BOSARIS toolkit", "Cllr / min Cllr calibration metrics"),
]
space(1)
for a, b in REFS:
    need(9)
    y = pdf.get_y()
    pdf.set_font("sans", "B", 8)
    pdf.set_text_color(*INK)
    pdf.set_xy(M, y)
    pdf.cell(42, 4.6, a)
    pdf.set_font("serif", "", 8.3)
    pdf.set_text_color(*INK2)
    pdf.set_xy(M + 42, y)
    pdf.multi_cell(W - 42, 4.6, b, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(0.7)

OUT = ("/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/research_logs/"
       "2026-08-10-dataset-quality-assessment-methods-and-plan.pdf")
pdf.output(OUT)
print("wrote", OUT, os.path.getsize(OUT), "bytes,", pdf.page_no(), "pages")
