#!/usr/bin/env python3
"""
render_analysis_pdf.py — convert SL_LANGUAGE_SPV_ANALYSIS.md to PDF via xelatex.

Pandoc is not installed in this environment, so this is a minimal,
project-specific markdown→LaTeX converter. It handles the constructs the
analysis doc actually uses:

  - ATX headings (# .. ####)
  - GitHub-style tables with header separator
  - Bullet and numbered lists
  - Inline code spans (backticks)
  - Fenced code blocks (```)
  - Bold (**) and italic (*)
  - Markdown links [text](url) including relative repo paths
  - Horizontal rules (---)
  - Blockquotes (>) — used by NESTED_ARCHITECTURE_FIXES.md style banner, not
    by the analysis doc itself, but supported for completeness

It does NOT handle: nested lists, HTML tags, footnotes, multi-line tables.
The analysis doc uses none of those.

Usage:
    python docs/render_analysis_pdf.py
"""

from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_MD = REPO_ROOT / "SL_LANGUAGE_SPV_ANALYSIS.md"
OUTPUT_TEX = REPO_ROOT / "SL_LANGUAGE_SPV_ANALYSIS.tex"
OUTPUT_PDF = REPO_ROOT / "SL_LANGUAGE_SPV_ANALYSIS.pdf"


# --------------------------------------------------------------------------
# LaTeX preamble
# --------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}

\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}[Scale=0.9]

\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{tabularx}
\usepackage{array}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{amssymb}     % for \checkmark, \square
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{fancyvrb}
\usepackage{parskip}
\usepackage{microtype}

\usepackage[colorlinks=true,
            linkcolor=blue!50!black,
            urlcolor=blue!50!black,
            citecolor=blue!50!black,
            pdfborder={0 0 0}]{hyperref}

\usepackage{titlesec}
\titleformat*{\section}{\Large\bfseries\sffamily}
\titleformat*{\subsection}{\large\bfseries\sffamily}
\titleformat*{\subsubsection}{\normalsize\bfseries\sffamily}

\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.15}

% Make tabularx columns vtop-aligned for long-cell tables
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}

\title{}
\author{}
\date{}

\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""


# --------------------------------------------------------------------------
# Inline transformations
# --------------------------------------------------------------------------

# Escape order matters. The first call replaces backslash, which the others
# rely on not creating new backslashes. Then the remaining specials.
def escape_latex(text: str) -> str:
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")
    return text


# Replace emoji / symbols with LaTeX-friendly equivalents BEFORE other
# processing. amssymb gives \checkmark and \square.
SYMBOL_MAP = {
    "✅": r"{\color{green!50!black}$\checkmark$}",
    "⬜": r"$\square$",
    "→": r"$\rightarrow$",
    "—": "---",       # em-dash (DejaVu has it but ascii is safer)
    "–": "--",        # en-dash
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "≈": r"$\approx$",
    "≠": r"$\neq$",
    "·": r"$\cdot$",
    "×": r"$\times$",
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "λ": r"$\lambda$",
    "σ": r"$\sigma$",
    "²": r"$^2$",
    "³": r"$^3$",
    "⁵": r"$^5$",
    "↔": r"$\leftrightarrow$",
    "…": r"\ldots{}",
    " ": " ",    # narrow no-break space
    " ": "~",    # no-break space
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "•": r"$\bullet$",
}


SYMBOL_PLACEHOLDER = "\x06SYM{}\x06"
_SYMBOL_PATTERN = re.compile("|".join(re.escape(k) for k in SYMBOL_MAP.keys()))


def extract_symbols(text: str) -> tuple[str, list[str]]:
    """Mask known unicode symbols with placeholders so subsequent
    escape_latex calls leave the eventual LaTeX form alone.

    Previously this was a straight string-replace before escape_latex,
    which then mangled the backslashes in the LaTeX-form replacements
    (``$\\rightarrow$`` became ``$\\textbackslash\\{\\}rightarrow$`` —
    rendering as literal text). Placeholder masking avoids that.
    """
    out: list[str] = []

    def repl(m: re.Match) -> str:
        out.append(SYMBOL_MAP[m.group(0)])
        return SYMBOL_PLACEHOLDER.format(len(out) - 1)

    return _SYMBOL_PATTERN.sub(repl, text), out


def reinsert_symbols(text: str, symbols: list[str]) -> str:
    for i, latex in enumerate(symbols):
        text = text.replace(SYMBOL_PLACEHOLDER.format(i), latex)
    return text


# Inline markdown → LaTeX. Operates on already-symbol-mapped text with
# code spans masked out as \x00CODEn\x00 placeholders.
def apply_inline(text: str) -> str:
    # Markdown links [text](url) — replace BEFORE escaping the rest so we
    # can handle the URL without escaping its slashes / hashes etc.
    def link_repl(m: re.Match) -> str:
        link_text = m.group(1)
        url = m.group(2)
        # Escape the visible text using the inline pipeline (recursive, but
        # link text is normally short and doesn't contain links).
        safe_text = escape_latex(link_text)
        # URLs need only #, %, \, and { } { escaped per hyperref.
        safe_url = url.replace("\\", r"\\").replace("#", r"\#").replace("%", r"\%")
        return r"\href{" + safe_url + "}{" + safe_text + "}"

    # Mask out link patterns first so their content isn't escaped twice.
    links: list[str] = []
    def link_mask(m: re.Match) -> str:
        links.append(link_repl(m))
        return f"\x01LINK{len(links)-1}\x01"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_mask, text)

    # Bold: **text** → \textbf{text}. Italic: *text* → \textit{text}.
    # Mark bold first to avoid italic eating the inner asterisks.
    bolds: list[str] = []
    def bold_mask(m: re.Match) -> str:
        bolds.append(m.group(1))
        return f"\x02BOLD{len(bolds)-1}\x02"

    text = re.sub(r"\*\*([^*]+)\*\*", bold_mask, text)

    italics: list[str] = []
    def italic_mask(m: re.Match) -> str:
        italics.append(m.group(1))
        return f"\x03ITAL{len(italics)-1}\x03"

    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", italic_mask, text)

    # Now escape LaTeX special chars in the remaining "plain" text.
    text = escape_latex(text)

    # Substitute back bold / italic with the inner content recursively
    # processed (one level deep; no nested bold-italic in this doc).
    for i, content in enumerate(bolds):
        text = text.replace(f"\x02BOLD{i}\x02", r"\textbf{" + apply_inline_no_links(content) + "}")
    for i, content in enumerate(italics):
        text = text.replace(f"\x03ITAL{i}\x03", r"\textit{" + apply_inline_no_links(content) + "}")
    for i, replacement in enumerate(links):
        text = text.replace(f"\x01LINK{i}\x01", replacement)

    return text


def apply_inline_no_links(text: str) -> str:
    """Used inside bold / italic spans — links were already extracted at the
    outer level."""
    # Escape only. Bold/italic nesting is not exercised by this doc.
    return escape_latex(text)


# --------------------------------------------------------------------------
# Code-span and code-block extraction
# --------------------------------------------------------------------------

CODE_SPAN_PLACEHOLDER = "\x04CODESPAN{}\x04"
CODE_BLOCK_PLACEHOLDER = "\x05CODEBLOCK{}\x05"


def extract_code(text: str) -> tuple[str, list[str], list[str]]:
    """Replace inline code and fenced code blocks with placeholders.
    Returns (masked_text, code_blocks, code_spans)."""

    code_blocks: list[str] = []
    code_spans: list[str] = []

    # Fenced code blocks first (multiline)
    def block_repl(m: re.Match) -> str:
        # m.group(1) = optional language; m.group(2) = body
        code_blocks.append(m.group(2))
        return CODE_BLOCK_PLACEHOLDER.format(len(code_blocks) - 1)

    text = re.sub(
        r"```([a-zA-Z0-9_+-]*)\n(.*?)\n```",
        block_repl,
        text,
        flags=re.DOTALL,
    )

    # Inline code spans
    def span_repl(m: re.Match) -> str:
        code_spans.append(m.group(1))
        return CODE_SPAN_PLACEHOLDER.format(len(code_spans) - 1)

    text = re.sub(r"`([^`\n]+)`", span_repl, text)

    return text, code_blocks, code_spans


def reinsert_code_spans(text: str, code_spans: list[str]) -> str:
    """After all other transformations, substitute back inline code spans as
    \\texttt{...} with each code-span's content properly escaped for
    \\texttt."""
    for i, content in enumerate(code_spans):
        # Inside \texttt{}, we need to escape LaTeX specials.
        escaped = escape_latex(content)
        replacement = r"\texttt{" + escaped + "}"
        text = text.replace(CODE_SPAN_PLACEHOLDER.format(i), replacement)
    return text


def reinsert_code_blocks(text: str, code_blocks: list[str]) -> str:
    """Substitute fenced code blocks back as Verbatim environments."""
    for i, content in enumerate(code_blocks):
        # Use fancyvrb's Verbatim environment with small font; no escaping
        # needed inside Verbatim.
        replacement = (
            "\n\\begin{Verbatim}[fontsize=\\small,frame=single,framesep=4pt]\n"
            + content
            + "\n\\end{Verbatim}\n"
        )
        text = text.replace(CODE_BLOCK_PLACEHOLDER.format(i), replacement)
    return text


# --------------------------------------------------------------------------
# Block-level conversion
# --------------------------------------------------------------------------

TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?(?:\s*\|\s*:?-+:?)+\s*\|?\s*$")


def split_table_row(line: str) -> list[str]:
    """Split a markdown table row by `|`, dropping leading/trailing empties."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def render_table(lines: list[str]) -> str:
    """Render a markdown table to a LaTeX tabularx. `lines` is a list of
    rows already identified as a contiguous table block (header + separator
    + body)."""
    # First row = header, second = separator, rest = body
    header_cells = split_table_row(lines[0])
    body_rows = [split_table_row(line) for line in lines[2:]]

    n_cols = len(header_cells)
    # Pad short rows to n_cols (defensive)
    for r in body_rows:
        while len(r) < n_cols:
            r.append("")

    # Use tabularx with Y (raggedright X) columns
    col_spec = "|" + "Y|" * n_cols

    out = []
    out.append("\\begin{center}")
    out.append("\\small")
    out.append("\\renewcommand{\\arraystretch}{1.25}")
    out.append("\\begin{tabularx}{\\textwidth}{" + col_spec + "}")
    out.append("\\hline")
    out.append(" & ".join(apply_inline(cell) for cell in header_cells) + " \\\\")
    out.append("\\hline")
    for row in body_rows:
        out.append(" & ".join(apply_inline(cell) for cell in row) + " \\\\")
        out.append("\\hline")
    out.append("\\end{tabularx}")
    out.append("\\end{center}")
    return "\n".join(out)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LIST_BULLET_RE = re.compile(r"^(\s*)[-*]\s+(.*)$")
LIST_NUMBERED_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
HR_RE = re.compile(r"^---+\s*$")
BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$")


def is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|") and line.rstrip().endswith("|") and "|" in line[1:]


def convert(md: str) -> str:
    """Top-level converter. Returns a LaTeX body (no preamble)."""

    # Pull code blocks and inline code out before anything else (they
    # must survive all subsequent processing untouched).
    md, code_blocks, code_spans = extract_code(md)

    # Mask known unicode symbols with placeholders so escape_latex
    # doesn't mangle their LaTeX-form backslashes downstream.
    md, symbols = extract_symbols(md)

    lines = md.split("\n")
    out: list[str] = []

    i = 0
    in_list = None  # None, "bullet", or "numbered"
    list_indent = 0

    def close_list() -> None:
        nonlocal in_list
        if in_list == "bullet":
            out.append("\\end{itemize}")
        elif in_list == "numbered":
            out.append("\\end{enumerate}")
        in_list = None

    while i < len(lines):
        line = lines[i]

        # 1) Heading
        m = HEADING_RE.match(line)
        if m:
            close_list()
            level = len(m.group(1))
            title = apply_inline(m.group(2))
            # # → section, ## → section, ### → subsection, #### → subsubsection
            if level == 1:
                out.append(f"\\section*{{{title}}}")
            elif level == 2:
                out.append(f"\\section*{{{title}}}")
            elif level == 3:
                out.append(f"\\subsection*{{{title}}}")
            else:
                out.append(f"\\subsubsection*{{{title}}}")
            i += 1
            continue

        # 2) Horizontal rule
        if HR_RE.match(line):
            close_list()
            out.append("\\vspace{0.5em}\\hrule\\vspace{0.5em}")
            i += 1
            continue

        # 3) Table (header row followed by separator row)
        if is_table_row(line) and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
            close_list()
            table_lines = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and is_table_row(lines[j]):
                table_lines.append(lines[j])
                j += 1
            out.append(render_table(table_lines))
            i = j
            continue

        # 4) Bullet list
        m_bullet = LIST_BULLET_RE.match(line)
        if m_bullet:
            indent_str, content = m_bullet.group(1), m_bullet.group(2)
            if in_list != "bullet":
                close_list()
                out.append("\\begin{itemize}[leftmargin=*,itemsep=2pt]")
                in_list = "bullet"
            out.append("  \\item " + apply_inline(content))
            i += 1
            continue

        # 5) Numbered list
        m_num = LIST_NUMBERED_RE.match(line)
        if m_num:
            indent_str, num, content = m_num.group(1), m_num.group(2), m_num.group(3)
            if in_list != "numbered":
                close_list()
                out.append("\\begin{enumerate}[leftmargin=*,itemsep=2pt]")
                in_list = "numbered"
            out.append("  \\item " + apply_inline(content))
            i += 1
            continue

        # 6) Blockquote
        m_bq = BLOCKQUOTE_RE.match(line)
        if m_bq:
            close_list()
            out.append("\\begin{quote}")
            out.append(apply_inline(m_bq.group(1)))
            i += 1
            # Continue grabbing subsequent blockquote lines
            while i < len(lines) and BLOCKQUOTE_RE.match(lines[i]):
                out.append(apply_inline(BLOCKQUOTE_RE.match(lines[i]).group(1)))
                i += 1
            out.append("\\end{quote}")
            continue

        # 7) Blank line
        if line.strip() == "":
            close_list()
            out.append("")
            i += 1
            continue

        # 8) Plain paragraph line (or continuation)
        # Collect contiguous non-blank, non-special lines
        para_lines = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if (
                nxt.strip() == ""
                or HEADING_RE.match(nxt)
                or HR_RE.match(nxt)
                or LIST_BULLET_RE.match(nxt)
                or LIST_NUMBERED_RE.match(nxt)
                or BLOCKQUOTE_RE.match(nxt)
                or is_table_row(nxt)
                or CODE_BLOCK_PLACEHOLDER.split("{")[0] in nxt
            ):
                break
            para_lines.append(nxt)
            i += 1

        para_text = " ".join(p.strip() for p in para_lines)
        # If the paragraph IS just a code-block placeholder, drop it through
        # (no apply_inline; let reinsert_code_blocks handle it).
        if re.fullmatch(r"\x05CODEBLOCK\d+\x05", para_text):
            close_list()
            out.append(para_text)
        else:
            close_list()
            out.append(apply_inline(para_text))

    close_list()
    body = "\n".join(out)

    # Reinsert in reverse order: symbols first (innermost), then code
    # spans, then code blocks. Each unmask step substitutes only its
    # own placeholder pattern.
    body = reinsert_symbols(body, symbols)
    body = reinsert_code_spans(body, code_spans)
    body = reinsert_code_blocks(body, code_blocks)

    return body


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    if not INPUT_MD.exists():
        print(f"[error] {INPUT_MD} not found", file=sys.stderr)
        return 1

    md_text = INPUT_MD.read_text(encoding="utf-8")
    tex_body = convert(md_text)
    OUTPUT_TEX.write_text(PREAMBLE + tex_body + POSTAMBLE, encoding="utf-8")
    print(f"[ok] wrote {OUTPUT_TEX} ({OUTPUT_TEX.stat().st_size:,} bytes)")

    # Compile with xelatex (twice for cross-references)
    cwd = REPO_ROOT
    for pass_num in (1, 2):
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode",
             "-halt-on-error", OUTPUT_TEX.name],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[error] xelatex pass {pass_num} failed (exit {result.returncode})",
                  file=sys.stderr)
            # Find the actual error line in the log
            tail = (result.stdout or "").splitlines()[-60:]
            print("\n".join(tail), file=sys.stderr)
            return result.returncode

    # Clean up intermediate files
    for ext in (".aux", ".log", ".out", ".toc"):
        p = REPO_ROOT / OUTPUT_TEX.with_suffix(ext).name
        if p.exists():
            p.unlink()

    if OUTPUT_PDF.exists():
        print(f"[ok] wrote {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size:,} bytes)")
        return 0
    print("[error] xelatex reported success but PDF not found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
