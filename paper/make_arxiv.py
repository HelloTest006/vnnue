"""Build the arXiv source bundle and verify it compiles the way arXiv does.

Usage (from paper/):
    modal run make_arxiv.py
Creates ../arxiv/vnnue-arxiv.tar.gz containing vnnue.tex, sec_*.tex, vnnue.bbl
and figures/*.png (no .bib: arXiv uses the shipped .bbl), then compiles that
exact bundle remotely with plain pdflatex (no bibtex) and prints ARXIV: OK/FAIL.
Run modal_tex.py first so that vnnue.bbl is current.
"""
import io
import pathlib
import re
import tarfile

import modal

app = modal.App("vnnue-arxiv-check")
image = modal.Image.debian_slim(python_version="3.12").apt_install(
    "texlive-latex-recommended", "texlive-latex-extra", "texlive-fonts-recommended",
    "texlive-science", "lmodern",
)


@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60)
def compile_bundle(tar_bytes: bytes) -> dict:
    import subprocess
    root = pathlib.Path("/work/arxiv")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        tf.extractall(root)
    out = ""
    for _ in range(3):  # arXiv runs pdflatex repeatedly; no bibtex
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "vnnue.tex"],
                           cwd=root, capture_output=True, text=True, timeout=600)
        out = r.stdout
    log = (root / "vnnue.log").read_text(errors="ignore")
    return {"rc": r.returncode, "log": log, "ok_pdf": (root / "vnnue.pdf").exists()}


@app.local_entrypoint()
def main():
    here = pathlib.Path(__file__).resolve().parent
    bbl = here / "vnnue.bbl"
    assert bbl.exists(), "run modal_tex.py first to produce vnnue.bbl"
    files = [here / "vnnue.tex", bbl] + sorted(here.glob("sec_*.tex")) + sorted((here / "figures").glob("*.png"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in files:
            tf.add(p, arcname=p.relative_to(here).as_posix())
    outdir = here.parent / "arxiv"
    outdir.mkdir(exist_ok=True)
    (outdir / "vnnue-arxiv.tar.gz").write_bytes(buf.getvalue())
    print("bundle:", [p.relative_to(here).as_posix() for p in files])
    res = compile_bundle.remote(buf.getvalue())
    log = res["log"]
    undef = re.findall(r"(Citation|Reference) `([^']+)' .*undefined", log)
    errors = [ln for ln in log.splitlines() if ln.startswith("!")]
    pages = re.findall(r"Output written on .*?\((\d+) pages", log)
    print("pages:", pages[-1] if pages else "?", "| errors:", errors[:10], "| undefined:", undef[:10])
    ok = res["ok_pdf"] and not errors and not undef
    print("written:", outdir / "vnnue-arxiv.tar.gz", f"({len(buf.getvalue())/1e6:.2f} MB)")
    print("ARXIV: OK" if ok else "ARXIV: FAIL")
