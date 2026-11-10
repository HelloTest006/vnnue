"""Compile the VNNUE paper on Modal (no local LaTeX).

Usage (from the paper/ directory):
    modal run modal_tex.py
Ships every .tex/.bib/.sty/.cls/.bst file in this directory plus figures/*.png,
runs latexmk -pdf (bibtex) remotely, writes vnnue.pdf locally and prints
warnings (undefined references/citations, overfull boxes) and TEX: OK / TEX: FAIL.
"""
import pathlib
import re

import modal

app = modal.App("vnnue-latex")

image = modal.Image.debian_slim(python_version="3.12").apt_install(
    "texlive-latex-recommended",
    "texlive-latex-extra",
    "texlive-fonts-recommended",
    "texlive-science",
    "texlive-bibtex-extra",
    "biber",
    "latexmk",
    "lmodern",
)

MAIN = "vnnue"


@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60)
def build(files: dict) -> dict:
    import subprocess

    root = pathlib.Path("/work/paper")
    root.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    r = subprocess.run(
        ["latexmk", "-pdf", "-bibtex", "-interaction=nonstopmode", "-halt-on-error",
         "-f", f"{MAIN}.tex"],
        cwd=root, capture_output=True, text=True, timeout=15 * 60,
    )
    log = (root / f"{MAIN}.log").read_text(errors="ignore") if (root / f"{MAIN}.log").exists() else ""
    blg = (root / f"{MAIN}.blg").read_text(errors="ignore") if (root / f"{MAIN}.blg").exists() else ""
    pdf = (root / f"{MAIN}.pdf").read_bytes() if (root / f"{MAIN}.pdf").exists() else b""
    bbl = (root / f"{MAIN}.bbl").read_bytes() if (root / f"{MAIN}.bbl").exists() else b""
    return {"rc": r.returncode, "out": (r.stdout + r.stderr)[-6000:], "log": log, "blg": blg, "pdf": pdf, "bbl": bbl}


@app.local_entrypoint()
def main():
    here = pathlib.Path(__file__).resolve().parent
    files = {}
    for ext in ("*.tex", "*.bib", "*.sty", "*.cls", "*.bst"):
        for p in here.glob(ext):
            files[p.name] = p.read_bytes()
    for p in (here / "figures").glob("*.png"):
        files[f"figures/{p.name}"] = p.read_bytes()
    res = build.remote(files)
    log = res["log"]
    if res["pdf"]:
        (here / f"{MAIN}.pdf").write_bytes(res["pdf"])
    if res.get("bbl"):
        (here / f"{MAIN}.bbl").write_bytes(res["bbl"])  # needed for arXiv
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' .*undefined", log)))
    undef_cit = sorted(set(re.findall(r"Citation `([^']+)' .*undefined", log)))
    overfull = re.findall(r"^Overfull \\hbox \(([\d.]+)pt too wide\)", log, re.M)
    big_overfull = [float(x) for x in overfull if float(x) > 10]
    errors = [ln for ln in log.splitlines() if ln.startswith("!")]
    pages = re.findall(r"Output written on .*?\((\d+) pages", log)
    bib_warn = [ln for ln in res["blg"].splitlines() if "Warning" in ln or "error" in ln.lower()]
    print("---- latexmk output tail ----")
    print(res["out"][-600:])
    print("---- summary ----")
    print("pages:", pages[-1] if pages else "?")
    print("errors:", errors[:20])
    lines = log.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("!"):
            print("  CONTEXT:", " | ".join(lines[max(0, i - 3):i + 4])[:600])
            break
    print("undefined references:", undef_ref)
    print("undefined citations:", undef_cit)
    print("bibtex warnings:", bib_warn[:20])
    for m in re.finditer(r"^Overfull \\hbox \(([\d.]+)pt too wide\) (.*)\n(.*)$", log, re.M):
        if float(m.group(1)) > 10:
            print("  OVERFULL", m.group(1), m.group(2)[:60], "::", m.group(3)[:100])
    print(f"overfull hboxes: {len(overfull)} (of which >10pt: {len(big_overfull)}; max {max([float(x) for x in overfull], default=0):.1f}pt)")
    ok = bool(res["pdf"]) and res["rc"] == 0 and not errors and not undef_ref and not undef_cit
    print("TEX: OK" if ok else "TEX: FAIL")
