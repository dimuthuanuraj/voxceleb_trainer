#!/usr/bin/env python3
"""One content model for the QC reports, rendered to HTML, Markdown and PDF.

The three formats are generated from the same block list, so they cannot drift
apart. Figures are written once as both PNG (for Markdown and PDF) and SVG (for
HTML), into <report_dir>/figures/, and referenced by key.

Block kinds:
    ("h2",      num, title)
    ("h3",      title)
    ("p",       text)                     markdown-ish inline: **bold**, `code`
    ("dek",     text)                     small muted lead-in
    ("kpis",    [(label, value, note, cls), ...])
    ("table",   [headers], [[cells], ...])
    ("fig",     key, caption)
    ("math",    label, tex)
    ("callout", kind, title, text)        kind in {"", "warn", "bad"}
    ("bullets", [text, ...])
"""
from __future__ import annotations

import html as _html
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qc_figures as F


# --------------------------------------------------------------------------- #
# figure store
# --------------------------------------------------------------------------- #
class Figures:
    """Collects figures, writes them as PNG + SVG, and hands back references."""

    def __init__(self, out_dir):
        self.dir = os.path.join(out_dir, "figures")
        os.makedirs(self.dir, exist_ok=True)
        self.svg = {}
        self.png = {}
        self._n = 0

    def add_svg(self, key, svg_text):
        """Store an SVG produced by qc_figures, and rasterise a PNG beside it."""
        self._n += 1
        stem = f"{self._n:02d}_{key}"
        with open(os.path.join(self.dir, stem + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(svg_text)
        self.svg[key] = svg_text
        png_path = os.path.join(self.dir, stem + ".png")
        _svg_to_png(svg_text, png_path)
        self.png[key] = png_path
        return key

    def add_figure(self, key, fig, dpi=190):
        """Store a live matplotlib figure directly (crisper PNG than converting)."""
        self._n += 1
        stem = f"{self._n:02d}_{key}"
        svg_buf = io.StringIO()
        fig.savefig(svg_buf, format="svg", bbox_inches="tight", pad_inches=.08)
        s = svg_buf.getvalue()
        s = s[s.index("<svg"):]
        with open(os.path.join(self.dir, stem + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(s)
        png_path = os.path.join(self.dir, stem + ".png")
        fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=.08, facecolor=fig.get_facecolor())
        plt.close(fig)
        self.svg[key] = s
        self.png[key] = png_path
        return key

    def math_png(self, tex, fontsize=14.0, dpi=340):
        """Render a formula at a TRUE point size, for placement at natural width.

        Returns (path, width_mm, height_mm). Rendering at `fontsize` points and
        `dpi`, then placing the image at width_px/dpi*25.4 mm, reproduces the
        glyphs at exactly `fontsize` points on the page. The base size is set
        above the 9.2 pt body deliberately: mathtext scales fraction parts to
        ~70 % and nested sub/superscripts to ~50 %, so a 14 pt base puts those
        inner terms at roughly body size, which is what makes the equation look
        native rather than shrunken.
        """
        self._n += 1
        stem = f"{self._n:02d}_eq"
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=fontsize, color=F.INK)
        png = os.path.join(self.dir, stem + ".png")
        fig.savefig(png, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=.05, transparent=True)
        plt.close(fig)
        from PIL import Image
        with Image.open(png) as im:
            w_mm = im.width / dpi * 25.4
            h_mm = im.height / dpi * 25.4
        return png, w_mm, h_mm


def _svg_to_png(svg_text, out_path):
    """Rasterise SVG without cairosvg: re-render is unavailable, so fall back to
    a matplotlib-drawn placeholder only if no converter exists."""
    try:
        import cairosvg  # optional
        cairosvg.svg2png(bytestring=svg_text.encode(), write_to=out_path,
                         output_width=1400)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# inline markup
# --------------------------------------------------------------------------- #
def _inline_html(t):
    t = _html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    return t


def _inline_md(t):
    return t


def _inline_plain(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    return t


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def render_html(meta, blocks, figs, css):
    b = [f'<title>{_html.escape(meta["title"])}</title>', f"<style>{css}</style>",
         '<div class="wrap">',
         '<header><div class="eyebrow">' + _html.escape(meta["eyebrow"]) + '</div>',
         f'<h1>{_html.escape(meta["h1"])}</h1>',
         f'<p class="sub">{_inline_html(meta["sub"])}</p>',
         '<div class="byline">' +
         "".join(f"<span>{_html.escape(x)}</span>" for x in meta["byline"]) +
         '</div></header>']
    for blk in blocks:
        k = blk[0]
        if k == "h2":
            b.append(f'<section><h2><span class="n">{blk[1]}</span>'
                     f'{_html.escape(blk[2])}</h2>')
        elif k == "h3":
            b.append(f"<h3>{_html.escape(blk[1])}</h3>")
        elif k == "p":
            b.append(f"<p>{_inline_html(blk[1])}</p>")
        elif k == "dek":
            b.append(f'<p class="dek">{_inline_html(blk[1])}</p>')
        elif k == "bullets":
            b.append("<ul>" + "".join(f"<li>{_inline_html(x)}</li>"
                                      for x in blk[1]) + "</ul>")
        elif k == "kpis":
            b.append('<div class="grid k4">')
            for lab, val, note, cls in blk[1]:
                b.append(f'<div class="kpi {cls}"><div class="lab">'
                         f'{_html.escape(lab)}</div><div class="val">'
                         f'{_html.escape(str(val))}</div><div class="note">'
                         f'{_html.escape(note)}</div></div>')
            b.append("</div>")
        elif k == "table":
            b.append('<div class="tw"><table><thead><tr>' +
                     "".join(f"<th>{_html.escape(h)}</th>" for h in blk[1]) +
                     "</tr></thead><tbody>")
            for row in blk[2]:
                b.append("<tr>" + "".join(
                    f'<td class="{"k" if i == 0 else "m" if _isnum(c) else ""}">'
                    f"{_inline_html(str(c))}</td>" for i, c in enumerate(row)) +
                    "</tr>")
            b.append("</tbody></table></div>")
        elif k == "fig":
            svg = figs.svg.get(blk[1], "")
            b.append(f'<figure>{svg}<figcaption>{_inline_html(blk[2])}'
                     f"</figcaption></figure>")
        elif k == "math":
            b.append(f'<div class="math"><div class="lab">'
                     f'{_html.escape(blk[1])}</div>{F.formula(blk[2])}</div>')
        elif k == "callout":
            kind, title, text = blk[1], blk[2], blk[3]
            b.append(f'<div class="callout {kind}"><strong>'
                     f"{_inline_html(title)}</strong> {_inline_html(text)}</div>")
        elif k == "endsection":
            b.append("</section>")
    b.append(f'<footer>{_inline_html(meta["footer"])}</footer></div>')
    return "\n".join(b)


def _isnum(c):
    return bool(re.fullmatch(r"[-+0-9.,%\s/–—]+", str(c)))


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def render_md(meta, blocks, figs):
    out = [f'# {meta["h1"]}', "",
           f'**{meta["eyebrow"]}**', "",
           _inline_md(meta["sub"]), "",
           " · ".join(f"`{x}`" for x in meta["byline"]), "", "---", ""]
    for blk in blocks:
        k = blk[0]
        if k == "h2":
            out += [f"## {blk[1]}. {blk[2]}", ""]
        elif k == "h3":
            out += [f"### {blk[1]}", ""]
        elif k == "p":
            out += [_inline_md(blk[1]), ""]
        elif k == "dek":
            out += [f"*{_inline_plain(blk[1])}*", ""]
        elif k == "bullets":
            out += [f"- {_inline_md(x)}" for x in blk[1]] + [""]
        elif k == "kpis":
            out += ["| Metric | Value | Note |", "|---|---|---|"]
            for lab, val, note, _ in blk[1]:
                out.append(f"| {lab} | **{val}** | {note} |")
            out.append("")
        elif k == "table":
            out.append("| " + " | ".join(blk[1]) + " |")
            out.append("|" + "|".join("---" for _ in blk[1]) + "|")
            for row in blk[2]:
                out.append("| " + " | ".join(str(c).replace("\n", " ")
                                             for c in row) + " |")
            out.append("")
        elif k == "fig":
            rel = os.path.join("figures", os.path.basename(
                figs.png.get(blk[1], "")))
            svg_rel = rel[:-4] + ".svg"
            if figs.png.get(blk[1]) and os.path.exists(figs.png[blk[1]]):
                out += [f"![{_inline_plain(blk[2])}]({rel})", ""]
            else:
                out += [f"![{_inline_plain(blk[2])}]({svg_rel})", ""]
            out += [f"*{_inline_plain(blk[2])}*", ""]
        elif k == "math":
            out += [f"**{blk[1]}**", "", f"$$\n{blk[2]}\n$$", ""]
        elif k == "callout":
            out += [f"> **{_inline_md(blk[2])}** {_inline_md(blk[3])}", ""]
    out += ["---", "", _inline_md(meta["footer"]), ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def render_pdf(meta, blocks, figs, out_path):
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    FONTS = os.path.join(os.path.dirname(matplotlib.__file__),
                         "mpl-data", "fonts", "ttf")
    INK, INK2, MUTED = (0x17, 0x22, 0x2D), (0x3A, 0x47, 0x53), (0x56, 0x64, 0x72)
    SIG, WARN, BAD = (0x00, 0x70, 0x7A), (0x8A, 0x5A, 0x12), (0x90, 0x38, 0x4A)
    RULE, RULESOFT = (0xD9, 0xDF, 0xE4), (0xE7, 0xEB, 0xEE)
    M = 16
    W = 210 - 2 * M

    def md_pdf(s):
        s = s.replace("`", "")
        s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"__\1__", s)
        return s.replace("--", "‑‑")

    class Doc(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("sans", "", 7)
            self.set_text_color(*MUTED)
            self.set_xy(M, 9)
            self.cell(W, 4, meta["running_head"])
            self.set_draw_color(*RULESOFT)
            self.set_line_width(.2)
            self.line(M, 14, 210 - M, 14)
            self.set_xy(M, self.t_margin)

        def footer(self):
            self.set_y(-13)
            self.set_font("mono", "", 7)
            self.set_text_color(*MUTED)
            self.cell(W, 4, str(self.page_no()), align="C")

    pdf = Doc("P", "mm", "A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_margins(M, 20, M)
    for fam, st, f in [
        ("serif", "", "DejaVuSerif.ttf"), ("serif", "B", "DejaVuSerif-Bold.ttf"),
        ("serif", "I", "DejaVuSerif-Italic.ttf"),
        ("serif", "BI", "DejaVuSerif-BoldItalic.ttf"),
        ("sans", "", "DejaVuSans.ttf"), ("sans", "B", "DejaVuSans-Bold.ttf"),
        ("sans", "I", "DejaVuSans-Oblique.ttf"),
        ("sans", "BI", "DejaVuSans-BoldOblique.ttf"),
        ("mono", "", "DejaVuSansMono.ttf"), ("mono", "B", "DejaVuSansMono-Bold.ttf"),
        ("mono", "I", "DejaVuSansMono-Oblique.ttf"),
        ("mono", "BI", "DejaVuSansMono-BoldOblique.ttf"),
    ]:
        pdf.add_font(fam, st, os.path.join(FONTS, f))
    pdf.set_title(meta["title"])
    pdf.set_author("SL_SPV")

    def space(h):
        pdf.set_y(pdf.get_y() + h)

    def need(h):
        if pdf.get_y() + h > 297 - 18:
            pdf.add_page()

    def rule_line(c=RULESOFT, w=.2):
        pdf.set_draw_color(*c)
        pdf.set_line_width(w)
        pdf.line(M, pdf.get_y(), 210 - M, pdf.get_y())

    # cover
    pdf.add_page()
    space(10)
    pdf.set_font("sans", "B", 7)
    pdf.set_text_color(*SIG)
    pdf.set_x(M)
    pdf.cell(W, 4, " ".join(meta["eyebrow"].upper()),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(3)
    pdf.set_font("sans", "B", 20)
    pdf.set_text_color(*INK)
    pdf.set_x(M)
    pdf.multi_cell(W, 8.8, meta["h1"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(2)
    pdf.set_font("serif", "", 10.5)
    pdf.set_text_color(*INK2)
    pdf.set_x(M)
    pdf.multi_cell(W * .92, 5.3, md_pdf(meta["sub"]),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT, markdown=True)
    space(3)
    rule_line(RULE, .3)
    space(2.4)
    pdf.set_font("mono", "", 7.3)
    pdf.set_text_color(*MUTED)
    pdf.set_x(M)
    pdf.cell(W, 4, "     ".join(meta["byline"]),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    space(4)

    for blk in blocks:
        k = blk[0]
        if k == "h2":
            need(26)
            space(4)
            pdf.set_draw_color(*INK)
            pdf.set_line_width(.5)
            pdf.line(M, pdf.get_y(), 210 - M, pdf.get_y())
            space(3)
            y = pdf.get_y()
            pdf.set_font("mono", "", 8)
            pdf.set_text_color(*SIG)
            pdf.set_xy(M, y)
            pdf.cell(12, 6, str(blk[1]))
            pdf.set_font("sans", "B", 12.5)
            pdf.set_text_color(*INK)
            pdf.set_xy(M + 12, y)
            pdf.multi_cell(W - 12, 6, blk[2], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            space(1)
        elif k == "h3":
            need(14)
            space(2)
            pdf.set_font("sans", "B", 9.8)
            pdf.set_text_color(*INK)
            pdf.set_x(M)
            pdf.multi_cell(W, 5, blk[1], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            space(.8)
        elif k in ("p", "dek"):
            pdf.set_font("serif", "I" if k == "dek" else "", 9.2)
            pdf.set_text_color(*(MUTED if k == "dek" else INK2))
            pdf.set_x(M)
            pdf.multi_cell(W, 4.7, md_pdf(blk[1]), new_x=XPos.LMARGIN,
                           new_y=YPos.NEXT, markdown=True)
            space(1.4)
        elif k == "bullets":
            for it in blk[1]:
                need(11)
                y = pdf.get_y()
                pdf.set_font("serif", "", 9)
                pdf.set_text_color(*SIG)
                pdf.set_xy(M + 1, y)
                pdf.cell(4, 4.5, "•")
                pdf.set_text_color(*INK2)
                pdf.set_xy(M + 5, y)
                pdf.multi_cell(W - 5, 4.5, md_pdf(it), new_x=XPos.LMARGIN,
                               new_y=YPos.NEXT, markdown=True)
                space(.7)
            space(1)
        elif k == "kpis":
            need(20)
            cols = 4
            cw = W / cols
            items = blk[1]
            for r0 in range(0, len(items), cols):
                row = items[r0:r0 + cols]
                y = pdf.get_y()
                for i, (lab, val, note, cls) in enumerate(row):
                    x = M + i * cw
                    pdf.set_xy(x, y)
                    pdf.set_font("sans", "B", 6.2)
                    pdf.set_text_color(*MUTED)
                    pdf.multi_cell(cw - 2, 3.4, lab.upper(),
                                   new_x=XPos.LEFT, new_y=YPos.NEXT)
                    pdf.set_x(x)
                    pdf.set_font("mono", "", 11)
                    col = {"good": SIG, "warn": WARN, "bad": BAD}.get(cls, INK)
                    pdf.set_text_color(*col)
                    pdf.multi_cell(cw - 2, 5, str(val), new_x=XPos.LEFT,
                                   new_y=YPos.NEXT)
                    if note:
                        pdf.set_x(x)
                        pdf.set_font("serif", "", 6.6)
                        pdf.set_text_color(*MUTED)
                        pdf.multi_cell(cw - 2, 3.2, note, new_x=XPos.LEFT,
                                       new_y=YPos.NEXT)
                pdf.set_y(y + 15)
            space(1.5)
        elif k == "table":
            headers, rows = blk[1], blk[2]
            n = len(headers)
            first = min(46, W * .34) if n > 3 else W / n
            rest = (W - first) / max(n - 1, 1)
            widths = [first] + [rest] * (n - 1)
            need(16)
            pdf.set_font("sans", "B", 6.1)
            pdf.set_text_color(*MUTED)
            # measure the wrapped header first: a fixed height made long headers
            # spill over the first data row
            hdr_lines = max(len(pdf.multi_cell(w - 1.5, 3.2, h.upper(),
                                               dry_run=True, output="LINES"))
                            for w, h in zip(widths, headers))
            y = pdf.get_y()
            x = M
            for w, h in zip(widths, headers):
                pdf.set_xy(x, y)
                pdf.multi_cell(w - 1.5, 3.2, h.upper(), new_x=XPos.RIGHT,
                               new_y=YPos.TOP)
                x += w
            pdf.set_y(y + hdr_lines * 3.2 + 1.4)
            rule_line(RULE)
            space(1.2)
            for row in rows:
                pdf.set_font("serif", "", 7.2)
                hs = []
                for w, c in zip(widths, row):
                    hs.append(len(pdf.multi_cell(w - 1.5, 3.8, md_pdf(str(c)),
                                                 dry_run=True, output="LINES",
                                                 markdown=True)))
                hh = max(hs) * 3.8 + 1.4
                need(hh + 3)
                y = pdf.get_y()
                x = M
                for i, (w, c) in enumerate(zip(widths, row)):
                    pdf.set_font("mono" if i == 0 else "serif",
                                 "B" if i == 0 else "", 6.8 if i == 0 else 7.2)
                    pdf.set_text_color(*(INK if i == 0 else INK2))
                    pdf.set_xy(x, y)
                    pdf.multi_cell(w - 1.5, 3.8, md_pdf(str(c)),
                                   new_x=XPos.RIGHT, new_y=YPos.TOP, markdown=True)
                    x += w
                pdf.set_y(y + hh)
                rule_line()
            space(2)
        elif k == "fig":
            png = figs.png.get(blk[1])
            if png and os.path.exists(png):
                from PIL import Image
                with Image.open(png) as im:
                    ar = im.height / im.width
                w = min(W, 168)
                h = w * ar
                if h > 200:
                    h = 200
                    w = h / ar
                need(h + 14)
                pdf.image(png, x=M + (W - w) / 2, w=w)
                space(1.5)
            pdf.set_font("serif", "I", 7.6)
            pdf.set_text_color(*MUTED)
            pdf.set_x(M)
            pdf.multi_cell(W, 3.8, _inline_plain(blk[2]),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            space(2.5)
        elif k == "math":
            png, w_mm, h_mm = figs.math_png(blk[2])
            avail = W - 6
            if w_mm > avail:                       # shrink only to avoid overflow
                h_mm *= avail / w_mm
                w_mm = avail
            need(h_mm + 12)
            pdf.set_font("sans", "B", 6.2)
            pdf.set_text_color(*MUTED)
            pdf.set_x(M)
            pdf.cell(W, 3.6, blk[1].upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            space(1.8)
            pdf.image(png, x=M + (W - w_mm) / 2, w=w_mm)   # centred display maths
            space(3.2)
        elif k == "callout":
            kind, title, text = blk[1], blk[2], blk[3]
            col = {"warn": WARN, "bad": BAD}.get(kind, SIG)
            need(18)
            space(1)
            y0 = pdf.get_y()
            pdf.set_x(M + 3.5)
            pdf.set_font("serif", "", 9)
            pdf.set_text_color(*INK2)
            pdf.multi_cell(W - 3.5, 4.6, md_pdf(f"**{title}** {text}"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT, markdown=True)
            y1 = pdf.get_y()
            pdf.set_draw_color(*col)
            pdf.set_line_width(.8)
            pdf.line(M, y0, M, y1)
            space(2)

    space(4)
    rule_line(RULE)
    space(1.5)
    pdf.set_font("serif", "", 7.4)
    pdf.set_text_color(*MUTED)
    pdf.set_x(M)
    pdf.multi_cell(W, 3.8, _inline_plain(meta["footer"]),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(out_path)
    return out_path
