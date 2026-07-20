#!/usr/bin/env python3
"""build_update.py — render a team update into a published page (+ PDF).

Workflow: owner drafts an update conversationally (text + pasted screenshots);
the master session writes updates/<date>_<slug>/update.md (front-matter: title,
date, tldr) + drops screenshots in updates/<date>_<slug>/figs/. Then:

    python tools/build_update.py updates/<date>_<slug> --publish --pdf
    python tools/build_console.py --publish   # to surface it on the console home

Image refs in update.md: `figs/x.png` copies alongside the published page;
any other relative path (e.g. `../../finetune_binary_v1/x.png`) is left as a
plain relative link, NOT copied — single source of truth for figures already
published elsewhere. Note the `../../` — a published update lives two levels
under PUBLISH_DEST (updates/<dirname>/index.html), one level deeper than a
normal result page.
"""
import argparse, datetime, pathlib, re, shutil, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PUBLISH_DEST = pathlib.Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen")
UPDATES_PUBLISH = PUBLISH_DEST / "updates"
CONSOLE_URL = "https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/index.html"

# .venv_console already has Markdown + PyYAML (installed for mkdocs). Reuse it
# rather than adding a new venv — this script is meant to be run with that
# interpreter: .venv_console/bin/python3 tools/build_update.py ...
try:
    import markdown
    import yaml
except ImportError:
    sys.exit("Run this with the console venv's python:\n"
             "  .venv_console/bin/python3 tools/build_update.py ...")

CHROME_CANDIDATES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]


# --------------------------------------------------------------- front matter ----
def parse_update_md(path):
    text = path.read_text()
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not m:
        sys.exit(f"{path}: must start with a --- front-matter block (title/date/tldr)")
    meta = yaml.safe_load(m.group(1)) or {}
    for k in ("title", "date", "tldr"):
        if k not in meta:
            sys.exit(f"{path}: front-matter missing required key '{k}'")
    return meta, m.group(2)


def resolve_image(src, update_dir, out_dir):
    """figs/* -> copy into out_dir, flattened, src rewritten to the bare filename.
    Anything else (http(s), or a relative path to an already-published figure)
    passes through unchanged — no copy, single source of truth."""
    if src.startswith(("http://", "https://", "data:")):
        return src
    if src.startswith("figs/"):
        rel = src[len("figs/"):]
        srcfile = update_dir / "figs" / rel
        if not srcfile.exists():
            sys.exit(f"referenced image not found: {srcfile}")
        destfile = out_dir / rel
        destfile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(srcfile, destfile)
        return rel
    return src  # existing published figure elsewhere — left as a relative link


IMG_TAG_RE = re.compile(r'<img([^>]*?)src="([^"]+)"([^>]*?)/?>')


def rewrite_images(html_body, update_dir, out_dir):
    def _sub(m):
        pre, src, post = m.groups()
        new_src = resolve_image(src, update_dir, out_dir)
        return f'<img{pre}src="{new_src}"{post}>'
    return IMG_TAG_RE.sub(_sub, html_body)


def wrap_figures(html_body):
    """A <p> containing exactly one <img alt="caption"> becomes a <figure> with
    a <figcaption> from the alt text (skipped if alt is empty)."""
    p_img_re = re.compile(r'<p>\s*(<img[^>]*>)\s*</p>')

    def _sub(m):
        img = m.group(1)
        alt_m = re.search(r'alt="([^"]*)"', img)
        alt = alt_m.group(1) if alt_m else ""
        cap = f'<figcaption>{alt}</figcaption>' if alt else ""
        return f'<figure>{img}{cap}</figure>'
    return p_img_re.sub(_sub, html_body)


def related_links_html(meta, dirname):
    related = meta.get("related") or []
    if not related:
        return ""
    items = []
    for entry in related:
        if isinstance(entry, dict):
            (slug, title), = entry.items()
        else:
            slug, title = entry, entry
        items.append(f'<a href="../../{slug}/index.html">{title}</a>')
    return " &middot; ".join(items)


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<div class="page">
{banner}
  <header>
    <h1>{title}</h1>
    <p class="meta">{date}</p>
    <p class="sub">{tldr}</p>
  </header>
{hero}
  <article>
{body}
  </article>

  <footer>
    <a href="{console_url}">&larr; Lightgen console</a>{related_sep}{related}
  </footer>
</div>
</body>
</html>
"""

CSS = """
  :root {
    --bg: #12151a; --panel: #171b22; --panel2: #1b2028; --border: #262c36;
    --text: #d8dde6; --muted: #8b96a5; --accent: #5cc8ff; --accent2: #ffb454;
    --good: #4ade80; --bad: #f87171;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 15px;
    line-height: 1.55; overflow-x: hidden; }
  .page { max-width: 1180px; margin: 0 auto; padding: 1.6rem 1.2rem 4rem; }
  h1 { font-size: 1.55rem; margin: 0 0 .3rem; letter-spacing: -.01em; }
  h2 { font-size: 1.15rem; margin: 1.6rem 0 .6rem; color: #fff; }
  h3 { font-size: .98rem; margin: 1.2rem 0 .5rem; color: #fff; }
  .meta { color: var(--muted); font-size: .78rem; margin: 0 0 .5rem; text-transform: uppercase;
          letter-spacing: .04em; }
  .sub { color: var(--muted); margin: 0 0 1.1rem; font-size: .95rem; max-width: 72ch; }
  p { margin: .6rem 0; }
  strong { color: #fff; }
  a { color: var(--accent); }
  code { background: #0d1014; border: 1px solid var(--border); border-radius: .25rem;
         padding: .05rem .3rem; font-size: .88em; color: var(--accent); }
  ul, ol { padding-left: 1.3rem; }
  li { margin: .3rem 0; }

  .banner { background: #7a1f1f; color: #fff; text-align: center; font-weight: 700;
            font-size: .85rem; letter-spacing: .04em; padding: .5rem 1rem;
            border-radius: .4rem; margin-bottom: 1rem; }

  .hero { margin: 1rem 0 1.6rem; }
  .hero img { width: 100%; max-height: 420px; object-fit: cover; border-radius: .5rem;
              border: 1px solid var(--border); display: block; }

  article { margin-top: .5rem; }
  article figure { margin: 1.1rem 0; background: var(--panel); border: 1px solid var(--border);
                   border-radius: .5rem; padding: .6rem; text-align: center; }
  article figure img { max-width: 100%; border-radius: .3rem; display: block; margin: 0 auto; }
  article figure figcaption { font-size: .78rem; color: var(--muted); margin-top: .5rem; }
  article img { max-width: 100%; border-radius: .3rem; }

  table { border-collapse: collapse; font-size: .85rem; width: 100%; margin: .8rem 0; }
  thead { background: var(--panel2); }
  th, td { padding: .5rem .7rem; border-bottom: 1px solid var(--border); text-align: left; }
  th { color: var(--muted); font-weight: 600; font-size: .72rem; text-transform: uppercase;
       letter-spacing: .03em; }
  .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }

  footer { margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
           color: var(--muted); font-size: .8rem; text-align: center; }
  footer a { color: var(--accent); }

  @media (max-width: 480px) {
    body { font-size: 14px; }
    .page { padding: 1.1rem .8rem 3rem; }
    h1 { font-size: 1.3rem; }
    .hero img { max-height: 260px; }
  }

  @media print {
    html, body { background: #fff; color: #111; font-size: 12px; }
    .page { max-width: 100%; padding: .6in .5in; }
    h1, h2, h3 { color: #111; }
    .meta, .sub { color: #444; }
    a { color: #1a4fa0; text-decoration: none; }
    code { background: #f0f0f0; color: #a03; border-color: #ccc; }
    article figure { background: #f7f7f7; border-color: #ccc; }
    article figure figcaption { color: #555; }
    th { background: #eee; color: #333; }
    th, td { border-color: #ccc; }
    footer { border-color: #ccc; color: #555; }
    .banner { background: #c33; }
    @page { margin: 0.6in; }
  }
"""


def build(update_dir: pathlib.Path, out_dir: pathlib.Path, template=False):
    md_path = update_dir / "update.md"
    if not md_path.exists():
        sys.exit(f"not found: {md_path}")
    meta, body_md = parse_update_md(md_path)

    if out_dir.exists():
        shutil.rmtree(out_dir)  # local staging copy only, never PUBLISH_DEST
    out_dir.mkdir(parents=True)

    body_html = markdown.markdown(
        body_md, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    body_html = rewrite_images(body_html, update_dir, out_dir)
    body_html = wrap_figures(body_html)

    hero_html = ""
    if hero := meta.get("hero"):
        hero_src = resolve_image(hero, update_dir, out_dir)
        hero_html = f'  <div class="hero"><img src="{hero_src}" alt=""></div>\n'

    banner_html = ""
    if meta.get("template"):
        banner_html = '  <div class="banner">TEMPLATE — not a real update</div>\n'

    related = related_links_html(meta, update_dir.name)
    html = PAGE_TEMPLATE.format(
        title=meta["title"], css=CSS, banner=banner_html, date=meta["date"],
        tldr=meta["tldr"], hero=hero_html, body=body_html,
        console_url=CONSOLE_URL, related_sep=" &middot; " if related else "",
        related=related,
    )
    (out_dir / "index.html").write_text(html)
    return out_dir / "index.html"


def publish(out_dir: pathlib.Path, dirname: str):
    dest = UPDATES_PUBLISH / dirname
    dest.mkdir(parents=True, exist_ok=True)
    # merge-copy, never rmtree the destination
    shutil.copytree(out_dir, dest, dirs_exist_ok=True)
    for p in [dest, *dest.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | (0o005 if p.is_dir() else 0o004))
        except OSError:
            pass
    return dest / "index.html"


def export_pdf(html_path: pathlib.Path, pdf_path: pathlib.Path):
    chrome = None
    for cand in CHROME_CANDIDATES:
        if shutil.which(cand):
            chrome = cand
            break
    if not chrome:
        sys.exit("no chrome/chromium binary found for --pdf "
                 "(tried: " + ", ".join(CHROME_CANDIDATES) + ")")
    cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox",
           f"--print-to-pdf={pdf_path}", "--print-to-pdf-no-header",
           "--no-pdf-header-footer", f"file://{html_path.resolve()}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not pdf_path.exists() or pdf_path.stat().st_size == 0:
        sys.exit(f"PDF export failed via {chrome}:\n{r.stdout}\n{r.stderr}")
    with open(pdf_path, "rb") as f:
        if f.read(5) != b"%PDF-":
            sys.exit(f"PDF export produced a non-PDF file: {pdf_path}")
    return chrome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("update_dir", help="e.g. updates/2026-07-05_my-update")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    update_dir = pathlib.Path(args.update_dir).resolve()
    dirname = update_dir.name
    stage_root = REPO / ".console_build" / "updates"
    out_dir = stage_root / dirname

    html_path = build(update_dir, out_dir)
    print(f"built: {html_path}")

    pdf_path = out_dir / f"{dirname}.pdf"
    if args.pdf:
        chrome = export_pdf(html_path, pdf_path)
        print(f"pdf: {pdf_path} ({pdf_path.stat().st_size} bytes, via {chrome})")

    if args.publish:
        dest_index = publish(out_dir, dirname)
        print(f"published: {dest_index}")
        print(f"URL: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/updates/{dirname}/index.html")
        if args.pdf:
            dest_pdf = UPDATES_PUBLISH / dirname / pdf_path.name
            print(f"PDF URL: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/updates/{dirname}/{pdf_path.name}")


if __name__ == "__main__":
    main()
