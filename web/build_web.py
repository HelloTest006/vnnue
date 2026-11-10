"""Turn the TeX4ht output (web/raw/vnnue.html) into the published reading page web/vnnue.html.

Run after `modal run paper/modal_html.py`:
    python web/build_web.py
"""
import html
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
raw = (HERE / "raw" / "vnnue.html").read_text(encoding="utf-8")

body = raw[raw.index("<body"):]
body = body[body.index(">") + 1:body.rindex("</body>")]

# drop TeX4ht's title block (we render our own masthead)
body = re.sub(r"<div class='maketitle'>.*?</div>\s*(?=<div class='abstract')", "", body, count=1, flags=re.S)
body = re.sub(r"<h2 class='titleHead'>.*?</h2>", "", body, flags=re.S)
body = re.sub(r"<div class='author'.*?</div>", "", body, flags=re.S)
body = re.sub(r"<div class='date'.*?</div>", "", body, flags=re.S)
# drop TeX4ht source-line comments and label comments
body = re.sub(r"<!--.*?-->", "", body, flags=re.S)


def squeeze(seg: str) -> str:
    seg = re.sub(r"[ \t]{2,}", " ", seg)
    seg = re.sub(r"\n\s*\n+", "\n", seg)
    return seg


parts = re.split(r"(<pre.*?</pre>)", body, flags=re.S)
body = "".join(p if p.startswith("<pre") else squeeze(p) for p in parts)

# Lean references under theorems -> marked kernel-checked lines
body = re.sub(
    r"<p class='noindent'>\s*<span class='ec-lmri-10'>Lean:\s*</span>",
    "<p class='leanref'><span class='lean-tag' title='Checked by the Lean 4 kernel'>Lean</span> ",
    body,
)
# Lean identifiers in monospace that start with VNNUE. -> link-free code style
body = re.sub(r"<span class='ec-lmtt-(?:9|10)'>(VNNUE\.[^<]+)</span>", r"<code class='lean'>\1</code>", body)

# wide content scrolls inside its own box
body = re.sub(r"(<table class='(?:tabular|longtable)'.*?</table>)", r"<div class='scroll'>\1</div>", body, flags=re.S)
body = re.sub(r"(<div class='algorithmic'>.*?</div>)", r"<div class='scroll'>\1</div>", body, flags=re.S)

# images: lazy, alt from caption is set by TeX4ht already
body = body.replace("<img ", "<img loading='lazy' ")

# table of contents from section headings
toc = []
for m in re.finditer(r"<h3 class='(sectionHead|likesectionHead)' id='([^']+)'>(.*?)</h3>", body, re.S):
    text = re.sub(r"<[^>]+>", "", m.group(3))
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    num, _, title = text.partition(" ")
    if not re.match(r"^[0-9A-C]$", num):
        num, title = "", text
    toc.append((m.group(2), num, title))
abstract_id = "abstract"
body = body.replace("<div class='abstract'", f"<div id='{abstract_id}' class='abstract'", 1)
toc_html = "\n".join(
    f"<li><a href='#{i}'><span class='toc-num'>{n}</span><span>{html.escape(t)}</span></a></li>"
    for i, n, t in [(abstract_id, "", "Abstract")] + toc
)

MACROS = r"""{
  Z: '\\mathbb{Z}', N: '\\mathbb{N}', R: '\\mathbb{R}',
  Zm: ['\\mathbb{Z}/2^{#1}', 1], one: '\\mathbf{1}',
  supp: '\\operatorname{supp}', colops: '\\operatorname{colOps}',
  dec: '\\operatorname{dec_{16}}', satop: '\\operatorname{sat_{16}}',
  fold: '\\operatorname{fold}', canon: '\\operatorname{canon}', clamp: '\\operatorname{clamp}',
  accV: '\\mathrm{acc}^{V}', accS: '\\mathrm{acc}^{S}', psum: 'p', symd: '\\mathbin{\\triangle}'
}"""

template = (HERE / "template.html").read_text(encoding="utf-8")
page = (template.replace("{{MACROS}}", MACROS)
        .replace("{{TOC}}", toc_html)
        .replace("{{BODY}}", body))
(HERE / "vnnue.html").write_text(page, encoding="utf-8")

# report anything MathJax would not know
math = re.findall(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", body, re.S)
cmds = set()
for a, b in math:
    cmds |= set(re.findall(r"\\([A-Za-z]+)", a or b))
print("sections:", len(toc), "| size:", len(page) // 1024, "KB | math commands:", len(cmds))
print("custom/unusual:", sorted(c for c in cmds if c in {"Z", "N", "R", "Zm", "one", "supp", "colops", "dec", "satop", "fold", "canon", "clamp", "accV", "accS", "psum", "symd", "lean", "lnm", "hd"}))

# Standalone copy for GitHub Pages (docs/): full document skeleton around the same page.
head, _, rest = page.partition("</style>")
standalone = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
    "<meta name=\"description\" content=\"VNNUE: an exact, Lean-verified update calculus and power-sum interaction channel for NNUE chess evaluation.\">\n"
    + head + "\nbody { margin: 0; }\nimg { max-width: 100%; }\n</style>\n</head>\n<body>\n" + rest + "\n</body>\n</html>\n"
)
docs = HERE.parent / "docs"
(docs / "figures").mkdir(parents=True, exist_ok=True)
(docs / "index.html").write_text(standalone, encoding="utf-8")
for png in (HERE / "figures").glob("*.png"):
    (docs / "figures" / png.name).write_bytes(png.read_bytes())
(docs / ".nojekyll").write_text("", encoding="utf-8")
print("standalone:", docs / "index.html")
