#!/usr/bin/env python3
"""build_roadmap.py — render ROADMAP.md into a live, auto-refreshing console page.

The roadmap is the real-time operational view (Now / Next / Waiting-on-you / thesis /
jobs / recent log). Source of truth is the repo's ROADMAP.md; this just wraps it in a
self-contained dark HTML page with a 2-min meta-refresh and an "updated" stamp, then
writes it straight to the NFS web dir (no mkdocs rebuild — fast, run it after every edit).

    python tools/build_roadmap.py            # write + report path
    python tools/build_roadmap.py --stamp "manual note"   # optional extra note

Cheap-update protocol: edit ROADMAP.md (a few lines), run this. That's it.
"""
import datetime, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "ROADMAP.md"
# NFS web dir the aspis server serves (repo is on local-scratch → can't symlink; write direct).
OUT = pathlib.Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen/roadmap/index.html")
URL = "https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/roadmap/index.html"

CSS = """
:root{--bg:#12151a;--panel:#171b22;--bd:#262c36;--tx:#d8dde6;--mut:#8b96a5;--acc:#5cc8ff;--warn:#ffb454}
*{box-sizing:border-box}html{font-size:17px}
body{margin:0;background:var(--bg);color:var(--tx);font:1rem/1.55 system-ui,-apple-system,Segoe UI,sans-serif;
 max-width:900px;margin:0 auto;padding:1.4rem 1.2rem 4rem}
h1{font-size:1.5rem;margin:.2rem 0 .1rem} h2{font-size:1.08rem;margin:1.8rem 0 .5rem;color:#fff;
 border-bottom:1px solid var(--bd);padding-bottom:.3rem}
.stamp{color:var(--mut);font-size:.82rem;margin-bottom:.6rem}
p{margin:.5rem 0} em{color:var(--mut)} strong{color:#fff}
code{background:#0d1014;border:1px solid var(--bd);border-radius:.25rem;padding:.05rem .3rem;font-size:.86em;color:var(--acc)}
ul,ol{margin:.4rem 0 .4rem 0;padding-left:1.4rem} li{margin:.28rem 0}
table{border-collapse:collapse;width:100%;font-size:.92rem;margin:.5rem 0}
th{text-align:left;color:var(--mut);font-size:.78rem;text-transform:uppercase;border-bottom:1px solid var(--bd);padding:.4rem .6rem}
td{padding:.4rem .6rem;border-bottom:1px solid #21262d}
a{color:var(--acc)}
@media(max-width:520px){body{padding:1rem .8rem 3rem}table{display:block;overflow-x:auto}}
/* pipeline strip */
.pipe{display:flex;flex-wrap:wrap;gap:.35rem;align-items:stretch;margin:.4rem 0 1rem}
.pnode{display:flex;flex-direction:column;justify-content:center;border:1px solid var(--bd);
 border-radius:.4rem;padding:.4rem .6rem;font-size:.82rem;min-width:0;position:relative}
.pnode .ic{font-size:.72rem;margin-bottom:.1rem;letter-spacing:.02em;text-transform:uppercase;opacity:.85}
.pnode.done{background:#12251a;border-color:#2ea04355;color:#b9e7c6}
.pnode.done .ic{color:#3fb950}
.pnode.doing{background:#2a1f0a;border-color:#ffb45488;color:#ffe0ad}
.pnode.doing .ic{color:#ffb454}
.pnode.todo{background:#161b22;color:var(--mut)} .pnode.todo .ic{color:#6e7681}
.pnode.wait{background:#0d1d2e;border-color:#1f6feb88;color:#bcd8ff} .pnode.wait .ic{color:#58a6ff}
.parr{align-self:center;color:#3a4250;font-size:.9rem}
.plegend{font-size:.74rem;color:var(--mut);margin:-.4rem 0 1rem}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}} .pnode.doing .ic{animation:pulse 1.6s infinite}
"""

ICON = {"done": "✓ done", "doing": "▶ now", "todo": "· todo", "wait": "⏸ you"}


def pipeline_html(md_text):
    """Extract a `## Pipeline` list of `- status: label` lines → colored strip.
    Returns (strip_html, md_without_that_section)."""
    import re
    m = re.search(r"\n## Pipeline\s*\n(.*?)(\n## |\Z)", md_text, re.S)
    if not m:
        return "", md_text
    nodes = re.findall(r"-\s*(done|doing|todo|wait)\s*:\s*(.+)", m.group(1))
    if not nodes:
        return "", md_text
    parts = []
    for i, (st, label) in enumerate(nodes):
        if i:
            parts.append('<span class=parr>→</span>')
        parts.append(f'<div class="pnode {st}"><span class=ic>{ICON[st]}</span>{label.strip()}</div>')
    strip = (f'<div class=pipe>{"".join(parts)}</div>'
             '<div class=plegend>✓ done · ▶ now · · todo · ⏸ waiting on you</div>')
    md_clean = md_text[:m.start()] + m.group(2) + md_text[m.end():]
    return strip, md_clean


def main():
    try:
        import markdown
    except ImportError:
        sys.exit("need `markdown` (use the console venv: .venv_console/bin/python tools/build_roadmap.py)")
    md = SRC.read_text()
    strip, md = pipeline_html(md)
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    # inject the pipeline strip right after the first </h1>
    if strip and "</h1>" in body:
        body = body.replace("</h1>", "</h1>\n" + strip, 1)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    extra = f" · {sys.argv[2]}" if len(sys.argv) > 2 and sys.argv[1] == "--stamp" else ""
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content=120>
<title>Lightgen roadmap</title><style>{CSS}</style></head><body>
<div class=stamp>page auto-refreshes every 2 min · rendered {now}{extra} ·
<a href="../index.html">↑ console</a></div>
{body}
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    OUT.chmod(0o644)
    print(f"wrote {OUT}\n{URL}")


if __name__ == "__main__":
    main()
