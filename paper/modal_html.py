"""Convert the VNNUE paper to HTML on Modal with TeX4ht (make4ht + MathJax output).

Usage (from the paper/ directory):
    modal run modal_html.py
Ships the .tex files, vnnue.bbl and figures; returns the generated HTML/CSS into
../web/raw/ for post-processing by build_web.py.
"""
import pathlib

import modal

app = modal.App("vnnue-html")
image = modal.Image.from_registry("texlive/texlive:TL2024-historic", add_python="3.12")


@app.function(image=image, cpu=2, memory=4096, timeout=30 * 60)
def convert(files: dict) -> dict:
    import subprocess
    root = pathlib.Path("/work/paper")
    root.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    # pdflatex passes first so .aux has all labels; the shipped .bbl provides the bibliography
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "vnnue.tex"], cwd=root,
                       capture_output=True, text=True, timeout=600)
    import shutil
    if shutil.which("make4ht"):
        cmd = ["make4ht", "-u", "-f", "html5", "vnnue.tex", "mathjax,fn-in"]
    else:
        cmd = ["htlatex", "vnnue.tex", "xhtml,mathjax,charset=utf-8,fn-in", " -cunihtf -utf8"]
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=1500)
    out = {}
    for p in root.iterdir():
        if p.suffix in (".html", ".css", ".svg", ".png") and p.is_file() and p.parent == root:
            out[p.name] = p.read_bytes()
    return {"rc": r.returncode, "log": (r.stdout + r.stderr)[-20000:], "files": out}


@app.local_entrypoint()
def main():
    here = pathlib.Path(__file__).resolve().parent
    files = {p.name: p.read_bytes() for p in [here / "vnnue.tex", here / "vnnue.bbl", *here.glob("sec_*.tex")]}
    for p in (here / "figures").glob("*.png"):
        files[f"figures/{p.name}"] = p.read_bytes()
    res = convert.remote(files)
    outdir = here.parent / "web" / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    for name, data in res["files"].items():
        (outdir / name).write_bytes(data)
    print(res["log"][-3000:])
    print("rc:", res["rc"], "files:", sorted(res["files"]))
