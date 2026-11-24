"""The paper's Fig. 1 for the README: a light and a dark PNG with a transparent background.

Needs pdflatex (with pgfplots) and PyMuPDF. Compiles figures/readme_fig1.tex twice, crops the page to the figure, and
renders it with an alpha channel.

Usage: python -B scripts/make_readme_fig1.py [--dest assets] [--dpi 300]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
MARGIN = 4.0  # pt around the figure


def compile_pdf(theme: str, out: Path) -> Path:
    job = f"readme_fig1_{theme}"
    source = "readme_fig1.tex" if theme == "light" else r"\def\readmedark{}\input{readme_fig1.tex}"
    for _ in range(2):  # pgfplots tables and labels settle on the second run
        run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={job}",
                              f"-output-directory={out}", source], cwd=FIGURES, capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit(f"pdflatex failed for {theme}; see {out / (job + '.log')}")
    return out / f"{job}.pdf"


def figure_box(page: fitz.Page) -> fitz.Rect:
    box = fitz.Rect()
    for d in page.get_drawings():
        box |= d["rect"]
    for b in page.get_text("blocks"):
        box |= fitz.Rect(b[:4])
    return fitz.Rect(box.x0 - MARGIN, box.y0 - MARGIN, box.x1 + MARGIN, box.y1 + MARGIN)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=ROOT / "assets")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    if shutil.which("pdflatex") is None:
        sys.exit("pdflatex not found")
    args.dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for theme in ("light", "dark"):
            with fitz.open(compile_pdf(theme, Path(tmp))) as doc:
                page = doc[0]
                pix = page.get_pixmap(dpi=args.dpi, clip=figure_box(page), alpha=True)
            dest = args.dest / f"concept_{theme}.png"
            pix.save(dest)
            print(f"{dest.relative_to(ROOT)}: {pix.width} x {pix.height} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
