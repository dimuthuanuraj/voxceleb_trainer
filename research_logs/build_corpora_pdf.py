#!/usr/bin/env python3
"""Build the Sinhala/Tamil SV corpora survey as a PDF.

Pure-Python: fpdf2 + the DejaVu family bundled inside matplotlib. No pandoc,
no TeX, no headless browser -- none of which are installed on this host.
"""
import os
import re
import matplotlib
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def md(s):
    """Normalise our markdown to the subset fpdf2 understands.

    fpdf2 knows **bold**, __italic__ and --underline--; it has no inline-code
    span, and a stray `--` (as in --session_from) would otherwise be eaten as an
    underline token.
    """
    s = s.replace("`", "")
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"__\1__", s)
    return s.replace("--", "‑‑")  # non-breaking hyphens

FONTS = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

# palette -- same tokens as the web version
INK     = (0x17, 0x22, 0x2D)
INK2    = (0x3A, 0x47, 0x53)
MUTED   = (0x56, 0x64, 0x72)
SIGNAL  = (0x00, 0x70, 0x7A)
CAUTION = (0x8A, 0x5A, 0x12)
REJECT  = (0x90, 0x38, 0x4A)
RULE    = (0xD9, 0xDF, 0xE4)
RULESOFT= (0xE7, 0xEB, 0xEE)
TINT    = (0xF4, 0xF5, 0xF6)
SIGTINT = (0xE2, 0xF1, 0xF2)

M = 18          # page margin, mm
W = 210 - 2 * M # content width


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("sans", "", 7)
        self.set_text_color(*MUTED)
        self.set_xy(M, 10)
        self.cell(W, 4, "Open Sinhala & Tamil Corpora for Speaker Verification",
                  align="L")
        self.set_draw_color(*RULESOFT)
        self.set_line_width(0.2)
        self.line(M, 15, 210 - M, 15)
        # FPDF leaves the cursor wherever header() finished -- put it back on
        # the top margin so body content never rides up onto the rule.
        self.set_xy(M, self.t_margin)

    def footer(self):
        self.set_y(-14)
        self.set_font("mono", "", 7)
        self.set_text_color(*MUTED)
        self.cell(W, 4, f"{self.page_no()}", align="C")


pdf = Doc(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(True, margin=20)
pdf.set_margins(M, 22, M)  # top margin clears the running header rule at y=15

for fam, style, f in [
    ("serif", "", "DejaVuSerif.ttf"), ("serif", "B", "DejaVuSerif-Bold.ttf"),
    ("serif", "I", "DejaVuSerif-Italic.ttf"),
    ("sans", "", "DejaVuSans.ttf"), ("sans", "B", "DejaVuSans-Bold.ttf"),
    ("mono", "", "DejaVuSansMono.ttf"), ("mono", "B", "DejaVuSansMono-Bold.ttf"),
]:
    pdf.add_font(fam, style, os.path.join(FONTS, f))

pdf.set_title("Open Sinhala & Tamil Corpora for Speaker Verification")
pdf.set_author("SL_SPV")


# ---------- primitives ----------
def space(h):
    pdf.set_y(pdf.get_y() + h)


def need(h):
    """Page-break if less than h mm remains."""
    if pdf.get_y() + h > 297 - 20:
        pdf.add_page()


def eyebrow(txt, color=SIGNAL):
    pdf.set_font("sans", "B", 7)
    pdf.set_text_color(*color)
    pdf.set_x(M)
    pdf.cell(W, 4, " ".join(txt.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def h2(num, txt):
    need(26)
    space(5)
    pdf.set_draw_color(*INK)
    pdf.set_line_width(0.5)
    pdf.line(M, pdf.get_y(), 210 - M, pdf.get_y())
    space(3.5)
    y = pdf.get_y()
    pdf.set_font("mono", "", 8)
    pdf.set_text_color(*SIGNAL)
    pdf.set_xy(M, y)
    pdf.cell(11, 6, num)
    pdf.set_font("sans", "B", 13)
    pdf.set_text_color(*INK)
    pdf.set_xy(M + 11, y)
    pdf.multi_cell(W - 11, 6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(1)


def dek(txt):
    pdf.set_font("serif", "I", 9)
    pdf.set_text_color(*MUTED)
    pdf.set_x(M)
    pdf.multi_cell(W, 4.6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(2)


def body(txt, size=9.5, color=INK2, lh=5.0, indent=0):
    pdf.set_font("serif", "", size)
    pdf.set_text_color(*color)
    pdf.set_x(M + indent)
    pdf.multi_cell(W - indent, lh, md(txt), new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                   markdown=True)
    space(1.6)


def rule(color=RULESOFT, w=0.2):
    pdf.set_draw_color(*color)
    pdf.set_line_width(w)
    pdf.line(M, pdf.get_y(), 210 - M, pdf.get_y())


# ---------- page 1: masthead ----------
pdf.add_page()
space(14)
eyebrow("SL_SPV · benchmark v1 sourcing")
space(3)
pdf.set_font("sans", "B", 22)
pdf.set_text_color(*INK)
pdf.set_x(M)
pdf.multi_cell(W, 9.5, "Open Sinhala & Tamil corpora\nfor speaker verification",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(3)
pdf.set_font("serif", "", 11)
pdf.set_text_color(*INK2)
pdf.set_x(M)
pdf.multi_cell(W * 0.82, 5.6,
               "Every openly-licensed corpus that could widen the si/ta benchmark "
               "— what each one actually gives you, and which three are worth "
               "the download.",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(4)
rule(RULE, 0.3)
space(2.5)
pdf.set_font("mono", "", 7.5)
pdf.set_text_color(*MUTED)
pdf.set_x(M)
pdf.cell(W, 4, "2026-08-10     survey — nothing downloaded yet     "
               "research_logs/2026-08-10-open-si-ta-sv-corpora.md",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(6)

# ---------- findings ----------
h2("00", "The short version")
dek("Three findings shape every recommendation below.")

FINDINGS = [
    ("FINDING 1", "Tamil is the binding constraint — and it is fixable",
     "49 speakers cannot resolve a 0.3 % EER delta. Two corpora take the Tamil "
     "pool past 750 speakers, and one of them ships ASV trial lists already built."),
    ("FINDING 2", "Sinhala has almost nothing new",
     "SLR52 *is* the open Sinhala corpus. The only genuinely new audio is SiTa "
     "(10 h, in-the-wild) and SLCeleb. Sinhala is absent from Common Voice entirely."),
    ("FINDING 3", "Every large Tamil corpus is Indian Tamil",
     "SLR65, SLR127, Kathbath, IndicVoices, Vaani — all recorded in India. "
     "Excellent training data; invalid for a Sri Lankan Tamil performance claim."),
]
for tag, title, txt in FINDINGS:
    need(24)
    y0 = pdf.get_y()
    pdf.set_fill_color(*TINT)
    pdf.rect(M, y0, W, 0.1, style="F")  # placeholder; block drawn below
    pdf.set_font("mono", "", 6.5)
    pdf.set_text_color(*SIGNAL)
    pdf.set_xy(M, y0 + 1)
    pdf.cell(W, 3.5, tag, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("sans", "B", 9.5)
    pdf.set_text_color(*INK)
    pdf.set_x(M)
    pdf.multi_cell(W, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(0.5)
    body(txt, size=9, lh=4.8)
    y1 = pdf.get_y()
    pdf.set_draw_color(*SIGNAL)
    pdf.set_line_width(0.7)
    pdf.line(M - 3, y0 + 0.5, M - 3, y1 - 1.6)
    space(1.5)

# ---------- chart ----------
need(78)
space(3)
pdf.set_font("sans", "B", 10)
pdf.set_text_color(*INK)
pdf.set_x(M)
pdf.cell(W, 5, "Speakers available per open corpus",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("serif", "", 8.5)
pdf.set_text_color(*MUTED)
pdf.set_x(M)
pdf.multi_cell(W, 4.2,
               "Distinct speakers with usable identity labels. Corpora whose "
               "speaker counts are undocumented are omitted rather than estimated.",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(3)

CHART = [
    ("TAMIL", [("Common Voice 26", 981, False),
               ("IISc-MILE · SLR127", 531, False),
               ("Kathbath ta", 226, False),
               ("SLR65", 49, True)]),
    ("SINHALA", [("SLR52", 478, True),
                 ("SLCeleb si", 150, False)]),
]
MAXV = 981
LABW, BARW = 40.0, 100.0

for grp, rows in CHART:
    need(10 + 7 * len(rows))
    pdf.set_font("sans", "B", 6.5)
    pdf.set_text_color(*MUTED)
    pdf.set_x(M)
    pdf.cell(W, 4, " ".join(grp), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    rule()
    space(2)
    for name, val, current in rows:
        y = pdf.get_y()
        pdf.set_font("mono", "", 7.5)
        pdf.set_text_color(*INK2)
        pdf.set_xy(M, y)
        pdf.cell(LABW, 5, name)
        w = BARW * val / MAXV
        pdf.set_fill_color(*SIGNAL)
        pdf.rect(M + LABW, y + 1.1, w, 3.4, style="F", round_corners=("TOP_RIGHT", "BOTTOM_RIGHT"), corner_radius=1.0)
        pdf.set_font("mono", "", 7.5)
        pdf.set_text_color(*INK2)
        pdf.set_xy(M + LABW + w + 2, y)
        pdf.cell(14, 5, f"{val}")
        if current:
            pdf.set_font("sans", "B", 5.5)
            pdf.set_text_color(*CAUTION)
            pdf.set_xy(M + LABW + w + 16, y)
            pdf.cell(20, 5, "I N   V 0")
        pdf.set_y(y + 5.6)
    space(1.5)

rule(RULE)
space(1.5)
pdf.set_font("mono", "", 6.5)
pdf.set_text_color(*MUTED)
pdf.set_x(M)
pdf.cell(W, 4, "0 — 981 speakers   ·   bars share one scale across both languages",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
space(2)
pdf.set_font("serif", "I", 8.5)
pdf.set_text_color(*MUTED)
pdf.set_x(M)
pdf.multi_cell(W, 4.2,
               "Sinhala has one open option and it is already in the benchmark. "
               "Tamil has three unused ones, each larger than what v0 runs on.",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ---------- corpus entries ----------
def entry(name, source, meta, licence, txt, lic_gated=False):
    need(30)
    space(2.5)
    rule()
    space(2.5)
    y0 = pdf.get_y()
    pdf.set_font("sans", "B", 10)
    pdf.set_text_color(*INK)
    pdf.set_xy(M, y0)
    pdf.cell(W, 5, name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("mono", "", 7)
    pdf.set_text_color(*MUTED)
    pdf.set_x(M)
    pdf.cell(W, 4, source, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(1.2)
    # metadata strip
    y = pdf.get_y()
    pdf.set_font("mono", "", 7.5)
    pdf.set_text_color(*INK2)
    pdf.set_xy(M, y)
    pdf.cell(W - 42, 5, meta)
    pdf.set_fill_color(*(SIGTINT if not lic_gated else (0xF6, 0xEE, 0xDE)))
    tw = pdf.get_string_width(licence) + 4
    pdf.rect(210 - M - tw, y + 0.4, tw, 4.4, style="F", round_corners=True, corner_radius=0.8)
    pdf.set_text_color(*(SIGNAL if not lic_gated else CAUTION))
    pdf.set_xy(210 - M - tw, y)
    pdf.cell(tw, 5, licence, align="C")
    pdf.set_y(y + 6)
    body(txt, size=9, lh=4.7)


pdf.add_page()
h2("01", "Use these")
dek("Open licence, speaker identity recoverable, enough speakers to move the numbers.")

entry("Kathbath / IndicSUPERB", "AI4Bharat",
      "ta (+11 Indic)   ·   226 Tamil speakers   ·   1,684 h all langs", "CC0",
      "The only corpus here with a **ready-made ASV protocol** — `valid_data.txt`, "
      "`test_known_data.txt`, `test_data.txt` per language. Speaker and gender sit in "
      "the filenames. Ships a **noisy test variant**, which is a free robustness axis "
      "for the ablation. Audio is m4a.")

entry("IISc-MILE Tamil ASR Corpus", "OpenSLR SLR127",
      "ta   ·   531 speakers   ·   ~150 h", "CC BY 2.0",
      "Biggest open Tamil speaker pool. Clean read speech, 16 kHz / 16-bit mono, "
      "train/test split already present. **Caveat:** the OpenSLR page does not "
      "document the speaker-ID convention — inspect filenames before budgeting "
      "the work.")

entry("Large Sinhala ASR training set", "OpenSLR SLR52",
      "si   ·   478 speakers (≥10 utts)   ·   ~185k utts", "CC BY-SA 4.0",
      "Already ingested and still the Sinhala backbone. Crowdsourced by Google in "
      "Sri Lanka; speaker IDs come from `utt_spk_text.tsv`. No gender, no session "
      "metadata.")

entry("SLCeleb", "ours · IEEE DataPort",
      "si + ta   ·   280 speakers   ·   34k utts", "CC BY 4.0",
      "Still the only source of multi-session, multi-genre, bilingual, "
      "Sri-Lankan-dialect speakers. Remains the v1 blocker — the audio is not on "
      "the local mounts.")

h2("02", "Worth adding, with caveats")
dek("Real value, but each carries a condition that changes how you can use it.")

entry("Common Voice 26.0", "Mozilla · released 2026-06-12",
      "ta only   ·   981 distinct voices   ·   425 h / 235 h validated", "CC0",
      "`client_id` is a pseudo-speaker ID and contributors record *across multiple "
      "days* — one of the few open sources of genuine **cross-session** Tamil "
      "trials. Gender is missing on ~66 % of clips, which constrains same-gender "
      "impostor sampling. No Sinhala.")

entry("IndicVoices / IndicVoices-R", "AI4Bharat · NeurIPS 2024",
      "ta (+21)   ·   10,496 speakers all langs   ·   1,704 h", "CC BY 4.0",
      "Spontaneous and read speech with rich per-speaker metadata — pitch, SNR, "
      "C50, demographics. Tamil subset also mirrored on Kaggle. Per-language speaker "
      "counts need checking before you plan around them.")

entry("Vaani", "ARTPARK-IISc × Google DeepMind",
      "ta (+104)   ·   158,441 speakers   ·   ~31,255 h raw", "CC BY 4.0",
      "Image-prompted spontaneous speech, downloadable **per district** — so you "
      "can pull Tamil Nadu only. The authors explicitly pitch it for speaker "
      "ID/verification. Far too large to be an eval set; treat it as a pretraining "
      "and AS-Norm cohort pool.")

entry("NISP", "IISc LEAP",
      "ta (+4 Indic +en)   ·   345 speakers   ·   ~4–5 min per speaker",
      "open · GitHub",
      "Small, but **every speaker records in both their mother tongue and English** "
      "— a real cross-lingual trial list without waiting on SLCeleb. Directly "
      "relevant to the P4 cross-lingual study.")

entry("SiTa", "Univ. of Moratuwa · CHiPSAL 2025",
      "si + ta   ·   1–10 spk/video   ·   602 min si, 121 min ta", "see repo",
      "The only new **Sri Lankan** in-the-wild audio — YouTube panel shows, "
      "debates, quizzes, code-mixed, real overlap. But the labels are *diarization "
      "turns*: speaker identity holds within a recording, not across them. Gives you "
      "conversational eval segments, not cross-recording trials, unless you link "
      "identities by hand.", lic_gated=True)

entry("SPRING-INX", "SPRING Lab, IIT Madras",
      "ta (+9)   ·   speaker count not stated   ·   ~2,000 h", "CC BY 4.0",
      "Large and legally clean, manually transcribed. Whether speakers are labelled "
      "well enough for SV needs verifying first.")

entry("Sinhala TTS", "OpenSLR SLR30 · Google",
      "si   ·   multi-speaker, count unconfirmed   ·   ~699 MB", "CC BY-SA 4.0",
      "UserID is embedded in the FileID, so speaker labels do exist. Almost certainly "
      "only a handful of speakers. Cheap to check, low expected yield.")

entry("Tamil multi-speaker set", "OpenSLR SLR65 · Google",
      "ta   ·   49 speakers (≥10 utts)   ·   4,286 utts", "CC BY-SA 4.0",
      "Already ingested. Keep it in the mix purely for continuity with the v0 numbers.")


# ---------- rejected ----------
h2("03", "Checked and rejected")
dek("Ruled out on purpose, so nobody re-searches these.")

REJECTED = [
    ("Common Voice Sinhala",
     "Does not exist. Not a size problem — the locale is absent from the corpus. "
     "Verified against the CV 26.0 locale list: 294 locales, no `si`."),
    ("VoxLingua107 si / ta",
     "67 h Sinhala and 51 h Tamil under CC BY 4.0, but the segments are cut by "
     "*automatic* diarization — no reliable speaker identity. Fine for SSL or "
     "domain pretraining of the frontend; invalid for trials."),
    ("FLEURS si_lk / ta_in",
     "Few speakers, no official speaker labels."),
    ("DISPLACE 2023 / 2024",
     "Conversational Tamil with speaker labels from IISc — the closest thing to "
     "what we actually want — but access requires a signed Terms & Conditions "
     "submission, so it is not openly available. Worth requesting separately."),
    ("Microsoft Speech Corpus (Indian)",
     "Tamil, Telugu, Gujarati. Research-only, non-commercial licence with mandatory "
     "attribution. Usable for a paper, not for the VoiceID product path."),
    ("IARPA Babel Tamil · LDC2017S13",
     "Conversational telephone Tamil, the classic SV-style data — but "
     "LDC-licensed and paid. Noted only for completeness."),
    ("Sinhala TTS sets",
     "SafnasKaldeen (HF), pnfo/sinhala-tts-dataset, SinhalaVITS — one to four "
     "speakers each. Useless for speaker verification."),
]
for name, why in REJECTED:
    need(18)
    space(1.5)
    rule()
    space(2)
    pdf.set_font("mono", "B", 8)
    pdf.set_text_color(*REJECT)
    pdf.set_x(M)
    pdf.cell(W, 4.5, name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(0.8)
    body(why, size=9, lh=4.7)


# ---------- ablation ----------
h2("04", "What this changes for the ablation")
dek("The feature-combination study is currently underpowered on Tamil.")
body("With 49 speakers the impostor pool is small enough that the EER confidence "
     "interval swamps the effects you are chasing — the P3 seed spread was "
     "already ±0.10 EER on Tamil. Widening the corpus is not a nice-to-have "
     "before the ablation; it is a precondition for the ablation meaning anything.")
space(2)

PLAN = [
    ("Kathbath Tamil first.", "CC0, ASV lists already exist, and the noisy variant "
     "doubles as a robustness condition. Lowest effort per unit of statistical power."),
    ("SLR127 second.", "Biggest speaker count, and it is the same read-speech domain "
     "as v0, so it drops into the existing protocol once the speaker-ID convention "
     "is confirmed."),
    ("Common Voice Tamil third,", "specifically to build the first cross-session "
     "Tamil trial list — `--session_from` finally has something to bind to."),
    ("SiTa and SLCeleb as the held-out Sri Lankan set.", "Never train on these."),
]
for i, (lead, rest) in enumerate(PLAN, 1):
    need(16)
    y = pdf.get_y()
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.25)
    pdf.circle(x=M, y=y + 2.4, radius=3.4)
    pdf.set_font("mono", "", 7.5)
    pdf.set_text_color(*SIGNAL)
    pdf.set_xy(M - 3.4, y + 0.4)
    pdf.cell(6.8, 4, str(i), align="C")
    pdf.set_xy(M + 8, y)
    pdf.set_font("serif", "", 9.5)
    pdf.set_text_color(*INK2)
    pdf.multi_cell(W - 8, 4.8, md(f"**{lead}** {rest}"),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT, markdown=True)
    space(2.2)

for title, txt in [
    ("Ingestion cost is small.",
     "`tools/ingest_openslr.py` is already structured as one `collect_<corpus>()` per "
     "source, so each new corpus is roughly a 25-line function plus a CLI flag. "
     "`tools/sl_dataprep.py` and the trial-list logic stay as they are."),
    ("The dialect split is a publishable axis, not just a caveat.",
     "Train on Indian Tamil (SLR127 + Kathbath), evaluate on Sri Lankan Tamil "
     "(SLCeleb / SiTa), report the gap. Nobody has published that number."),
]:
    need(22)
    space(1)
    y0 = pdf.get_y()
    pdf.set_x(M + 4)
    pdf.set_font("serif", "", 9.5)
    pdf.set_text_color(*INK2)
    pdf.multi_cell(W - 4, 4.8, md(f"**{title}** {txt}"),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT, markdown=True)
    y1 = pdf.get_y()
    pdf.set_draw_color(*SIGNAL)
    pdf.set_line_width(0.8)
    pdf.line(M, y0, M, y1)
    space(2.5)


# ---------- sources ----------
h2("05", "Sources")
SOURCES = [
    ("OpenSLR index (SLR30, 52, 65, 127)", "openslr.org/resources.php"),
    ("IISc-MILE Tamil ASR", "openslr.org/127"),
    ("IndicSUPERB / Kathbath", "github.com/AI4Bharat/IndicSUPERB · arXiv:2208.11761"),
    ("Kathbath data", "huggingface.co/datasets/ai4bharat/Kathbath"),
    ("IndicVoices-R", "github.com/AI4Bharat/IndicVoices-R · arXiv:2409.05356"),
    ("Vaani", "vaani.iisc.ac.in · huggingface.co/datasets/ARTPARK-IISc/Vaani"),
    ("Common Voice stats", "github.com/common-voice/cv-dataset — cv-corpus-26.0-2026-06-12.json"),
    ("NISP", "github.com/iiscleap/NISP-Dataset · arXiv:2007.06021"),
    ("SiTa", "aclanthology.org/2025.chipsal-1.8 · github.com/SiTa-SpeakerDiarization/SiTa"),
    ("SPRING-INX", "arXiv:2310.14654"),
    ("VoxLingua107", "cs.taltech.ee/staff/tanel.alumae/data/voxlingua107"),
    ("DISPLACE", "displace2024.github.io"),
    ("Microsoft Speech Corpus", "microsoft.com/en-us/download/details.aspx?id=105292"),
]
space(1)
for label, url in SOURCES:
    # label + url must move to a new page together, never straddle the break
    with pdf.unbreakable():
        pdf.set_font("serif", "", 8.5)
        pdf.set_text_color(*INK2)
        pdf.set_x(M)
        pdf.cell(58, 4.6, label)
        pdf.set_font("mono", "", 7)
        pdf.set_text_color(*SIGNAL)
        pdf.set_x(M + 58)
        pdf.multi_cell(W - 58, 4.6, url, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(0.6)

OUT = "/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/research_logs/2026-08-10-open-si-ta-sv-corpora.pdf"
pdf.output(OUT)
print("wrote", OUT, os.path.getsize(OUT), "bytes,", pdf.page_no(), "pages")
