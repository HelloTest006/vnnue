"""Check VNNUE Lean files on a Modal CPU instance with Mathlib v4.35.0 prebuilt.

Usage (from the lean/ directory):
    modal run modal_check.py                      # check every VNNUE/**/*.lean + VNNUE.lean
    modal run modal_check.py --files VNNUE/Basic.lean,VNNUE/Accumulator.lean
Each file is compiled in dependency order via `lake build` of the module targets.
Output: compiler messages, and a final line `RESULT: OK` or `RESULT: FAIL`.
Also reports any `sorry`/`admit`/`axiom` usage.
"""
import modal, pathlib

MATHLIB_REV = "v4.35.0"
app = modal.App("vnnue-lean-check")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git", "ca-certificates")
    .run_commands(
        "curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"
        " | sh -s -- -y --default-toolchain none",
        f"git clone --depth 1 --branch {MATHLIB_REV} https://github.com/leanprover-community/mathlib4.git /opt/mathlib",
        "cd /opt/mathlib && /root/.elan/bin/lake exe cache get",
        "cd /opt/mathlib && /root/.elan/bin/lake env /root/.elan/bin/lean --version",
    )
)

LAKEFILE = """name = "vnnue"
defaultTargets = ["VNNUE"]

[[require]]
name = "mathlib"
path = "/opt/mathlib"

[[lean_lib]]
name = "VNNUE"
"""


@app.function(image=image, cpu=8, memory=16384, timeout=60 * 60)
def check(files: dict, targets: list) -> dict:
    import os, subprocess, re
    root = pathlib.Path("/work/vnnue")
    root.mkdir(parents=True, exist_ok=True)
    (root / "lakefile.toml").write_text(LAKEFILE)
    (root / "lean-toolchain").write_text(open("/opt/mathlib/lean-toolchain").read())
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    env = dict(os.environ, PATH="/root/.elan/bin:" + os.environ["PATH"])
    r = subprocess.run(["lake", "build", *targets], cwd=root, env=env,
                       capture_output=True, text=True, timeout=55 * 60)
    out = r.stdout + r.stderr
    # axiom audit: print axioms for every theorem in the checked modules
    audit = ""
    if r.returncode == 0:
        mods = [t for t in targets]
        lines = [f"import {m}" for m in mods] + ["", "open Lean in", "#eval show Lean.Elab.Command.CommandElabM Unit from pure ()"]
        audit_src = "\n".join(f"import {m}" for m in mods)
        decls = []
        audit_mods = set(mods)
        if "VNNUE" in audit_mods:
            audit_mods |= {r[:-5].replace("/", ".") for r in files}
        for rel, src in files.items():
            if rel[:-5].replace("/", ".") not in audit_mods:
                continue
            ns = None
            for m in re.finditer(r"^namespace\s+(\S+)|^end\s+(\S+)|^(?:@\[[^\]]*\]\s*)?(?:private\s+)?(?:theorem|lemma)\s+(\S+)", src, re.M):
                if m.group(1): ns = m.group(1)
                elif m.group(2) and ns == m.group(2): ns = None
                elif m.group(3):
                    decls.append((ns + "." if ns else "") + m.group(3))
        audit_src += "\n" + "\n".join(f"#print axioms {d}" for d in decls) + "\n"
        (root / "Audit.lean").write_text(audit_src)
        a = subprocess.run(["lake", "env", "lean", "Audit.lean"], cwd=root, env=env,
                           capture_output=True, text=True, timeout=20 * 60)
        audit = a.stdout + a.stderr
    return {"rc": r.returncode, "out": out[-60000:], "audit": audit[-60000:]}


@app.local_entrypoint()
def main(files: str = ""):
    here = pathlib.Path(".")
    if files:
        paths = [here / f for f in files.split(",")]
    else:
        paths = sorted(here.glob("VNNUE/**/*.lean"))
        if (here / "VNNUE.lean").exists():
            paths.append(here / "VNNUE.lean")
    # always ship every VNNUE source so imports resolve
    payload = {str(p.as_posix()): p.read_text(encoding="utf-8") for p in sorted(here.glob("VNNUE/**/*.lean"))}
    if (here / "VNNUE.lean").exists():
        payload["VNNUE.lean"] = (here / "VNNUE.lean").read_text(encoding="utf-8")
    targets = [p.as_posix()[:-5].replace("/", ".") for p in paths]
    res = check.remote(payload, targets)
    print(res["out"])
    sorry = [ln for ln in (res["out"] + res["audit"]).splitlines() if "sorry" in ln.lower()]
    print("---- AXIOM AUDIT ----")
    print(res["audit"])
    bad_ax = [ln for ln in res["audit"].splitlines() if "sorryAx" in ln]
    ok = res["rc"] == 0 and not bad_ax and not any("declaration uses 'sorry'" in s for s in sorry)
    print("RESULT: OK" if ok else "RESULT: FAIL")
