"""xgpage.py — small, dependency-free (stdlib only) HTML component library for
lightgen's published result pages.

Extracted 2026-07-06 from build_finetune_page.py (the first page built with the
outline/scrollspy + Medium-style preview/expand mechanic) so the design system
(web/assets/theme.css + web/assets/ui.js + web/assets/katex/) has ONE Python-side
API instead of being copy-pasted per page. Pages import this module and call its
functions with THEIR OWN verified numbers/captions — xgpage only assembles
markup, it never invents content.

Data contracts are documented per-function below. General shape: every component
function returns a plain HTML string (never writes files itself); `page()` is the
only function that assembles a full document and is normally called once, last.

Usage sketch:

    import xgpage as lp

    body = []
    body.append(lp.section("overview", None, "Overview", body_html=OVERVIEW_HTML))
    body.append(lp.section("results", 2, "Results: ...", takeaway="...",
                            body_html=RESULTS_HTML, preview_rem=36.5))
    ...
    html = lp.page(
        title="...",
        header_html=lp.header("Page H1", "sub text with <a> allowed"),
        body_sections=body,
        outline_entries=[
            {"id": "overview", "label": "Overview"},
            {"id": "results", "label": "Results"},
            {"id": "method", "label": "Method", "sub": [
                {"id": "tensor-inventory", "label": "Tensor inventory"},
                {"id": "loss-math", "label": "Loss math"},
            ]},
        ],
        needs_katex=True,
    )
    open(out_path, "w").write(html)

Publishing: theme.css/ui.js/katex/ live in web/assets/ (repo source of truth) and
are merge-copied (never rmtree) to PUBLISH_DEST/assets/. Individual page dirs are
published as before (aspis-publish.sh, which fully replaces just that one page's
own directory). `assets_rel` (default "../assets", correct for a normal one-level
page dir like PUBLISH_DEST/<slug>/index.html) controls the path pages use to reach
the shared assets — pass "../../assets" for a page nested one level deeper (e.g.
PUBLISH_DEST/updates/<dir>/index.html).

IMPORTANT for any page that will move directories after being built (the
_preview/<name>/ -> <name>/ promotion workflow): a RELATIVE assets_rel silently
breaks on promotion, because _preview/<name>/ and <name>/ are NOT the same
nesting depth under the web root (_preview adds one level) even though they
look superficially similar. Pass a SITE-ABSOLUTE assets_rel instead (e.g.
"/projects/<org>/<user>/<project>/assets") for any page that will be `mv`'d —
it resolves identically regardless of which directory ends up serving it. Same
logic applies to any image/figure src that reaches into ANOTHER page's
directory: prefer copying those images into this page's own directory (self
-contained) over a relative reference, since promotion can also REPLACE the
directory being referenced. See xgpage SKILL.md rule 11 for the incident this
was learned from.
"""

import hashlib as _hashlib
import html as _html
import os


def _file_hash8(path):
    """8-char md5 prefix of a file's content, for cache-busting query strings.
    Returns "0" (a harmless, stable fallback) if the file can't be read, so a
    missing/misconfigured assets_dir degrades to a constant query string rather
    than crashing the whole page build."""
    try:
        with open(path, "rb") as f:
            return _hashlib.md5(f.read()).hexdigest()[:8]
    except OSError:
        return "0"


# ---------------------------------------------------------------- page assembly
def page(*, title, body_sections, header_html="", nav_html="", outline_entries=None,
         assets_rel="../assets", assets_dir=None, needs_katex=False, extra_head="", extra_body_end="",
         theme=None, tree_html="", nav_title="", version_slot="", wide=False):
    """Assemble a full HTML document.

    - title: <title> text.
    - header_html: raw HTML inserted before the sections (typically from header()).
    - nav_html: raw HTML inserted at the very top of <body>, OUTSIDE .layout/.page
      (typically from nav_tabs()/nav_subtabs()) — a cross-page nav shell that should
      span the full viewport width rather than being constrained to .page's max-width.
      Added 2026-07-10 (somages console v11). Empty by default — fully backward
      compatible with pages built before this parameter existed.
    - body_sections: list of already-built section HTML strings, concatenated in order.
    - outline_entries: None -> no sidebar, plain single-column `.page` (still gets
      ui.js, which no-ops harmlessly with no .outline/.collapsible in the DOM).
      Otherwise a list of dicts: {"id", "label", "sub": [{"id","label"}, ...]} (sub optional).
      Renders the `.outline` nav wrapped in `.layout`, with scrollspy wired to every
      id mentioned (top-level and sub) via data-spy / data-spy-link, and click-to-
      expand via data-target (top-level id, or the top-level id owning a sub-entry).
    - assets_rel: relative path prefix to web/assets/ from this page's own directory.
    - assets_dir: optional ABSOLUTE path to the web/assets/ directory on disk. When
      given, theme.css and ui.js are cache-busted with an 8-char md5 of their own
      file content (?v=<hash>) — the same convention this system's own SKILL.md
      already prescribes for figure images, just never applied to the shared assets
      themselves (found the gap 2026-07-07: a Playwright verification pass kept
      re-serving a stale theme.css mid-session purely because the link has no query
      string to bust on). Omit (default None) for no cache-busting — fully
      backward compatible with pages built before this parameter existed.
    - needs_katex: include the katex.min.css link + runtime render-loop script.
    - extra_head / extra_body_end: raw HTML spliced into <head> / just before </body>
      (after ui.js), for page-specific one-offs.
    - theme: None (default) -> the original dark v1 look, byte-identical to pages
      built before this parameter existed. "v2" -> the editorial design language
      (2026-07-14): additionally links assets/theme2.css (light default + dark via
      prefers-color-scheme), loads assets/xg2.js (chart tooltips, compare sliders),
      and sets <body class="xg2">. v1 components still render inside a v2 page
      (theme2.css re-skins them); the v2-only components below (statband,
      hero_header, fig, hbar_chart, compare_slider, ...) REQUIRE theme="v2".
      "v3" -> the "workspace" design language (2026-07-16): v2's editorial style
      at v1's density inside a three-column shell (left page tree, wider/compacter
      centered content column, right per-page outline). Links theme2.css AND
      theme3.css, loads xg2.js AND xg3.js, sets <body class="xg2 xg3"> so the
      ENTIRE v2 component library renders unchanged and theme3.css only overrides
      density and adds the shell. D11/D12 (viewport-relative centering) are
      superseded on v3: the content column is fixed by the grid, so figures and
      prose center within THAT column (container-relative, via the same
      _center_frag margin:auto mechanism), not the viewport. Requires the
      workspace chrome params below.
    - tree_html: v3 only. Prebuilt left workspace page-tree nav (from v3_tree()).
      Same on every page of a workspace; the active leaf marks the current page.
    - nav_title: v3 only. Short title shown in the mobile top bar next to the
      tree-toggle button (the full <title> is too long for that bar).
    - version_slot: v3 only. Prebuilt version-slot HTML (from
      v3_version_slot()), floated at the top-right of the content column on
      the eyebrow band. Empty (default) renders no slot.
    - wide: v3 only (2026-07-17, console pages-table rework). Full-width
      variant for database/table pages: the shell drops the outline track
      and the content column fills the space between the rails
      (.v3-shell.v3-wide + .page.page-wide in theme3.css). The 820px
      measure is a PROSE law (D3); a full-page table has no prose measure
      to keep. outline_entries is ignored in this variant.
    """
    theme_v = f"?v={_file_hash8(os.path.join(assets_dir, 'theme.css'))}" if assets_dir else ""
    ui_v = f"?v={_file_hash8(os.path.join(assets_dir, 'ui.js'))}" if assets_dir else ""
    head_katex = f'<link rel="stylesheet" href="{assets_rel}/katex/katex.min.css">\n' if needs_katex else ""

    def _av(name):
        """Cache-bust query for a shared asset filename (empty if no assets_dir)."""
        return f"?v={_file_hash8(os.path.join(assets_dir, name))}" if assets_dir else ""

    # v2 and v3 both build on theme2.css/xg2.js (the editorial palette + component
    # library); v3 layers theme3.css/xg3.js (the workspace shell + density) ON TOP
    # and keeps <body class="xg2 xg3"> so every v2 component renders unchanged.
    head_theme2 = ""
    body_cls = ""
    xg2_script = ""
    if theme in ("v2", "v3"):
        head_theme2 = f'<link rel="stylesheet" href="{assets_rel}/theme2.css{_av("theme2.css")}">\n'
        xg2_script = f'<script src="{assets_rel}/xg2.js{_av("xg2.js")}"></script>\n'
        body_cls = ' class="xg2"'
    if theme == "v3":
        head_theme2 += f'<link rel="stylesheet" href="{assets_rel}/theme3.css{_av("theme3.css")}">\n'
        xg2_script += f'<script src="{assets_rel}/xg3.js{_av("xg3.js")}"></script>\n'
        body_cls = ' class="xg2 xg3"'
    body_html = "\n".join(body_sections)

    if theme == "v3":
        outline_html = "" if wide else (
            _v3_outline_nav(outline_entries) if outline_entries else "")
        topbar = _v3_topbar(nav_title)
        shell_cls = "v3-shell v3-wide" if wide else "v3-shell"
        page_cls = "page page-wide" if wide else "page"
        content = (f'{topbar}\n<div class="{shell_cls}">\n{tree_html}\n'
                   f'<div class="v3-scrim" id="v3-scrim"></div>\n'
                   f'<main class="v3-main"><div class="{page_cls}">\n{version_slot}{header_html}\n{body_html}\n</div></main>\n'
                   f'{outline_html}\n</div>')
    elif outline_entries:
        nav = _outline_nav(outline_entries)
        content = f'<div class="layout">\n{nav}\n<div class="page">\n{header_html}\n{body_html}\n</div>\n</div>'
    else:
        content = f'<div class="page">\n{header_html}\n{body_html}\n</div>'

    katex_script = _katex_runtime_script(assets_rel) if needs_katex else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="{assets_rel}/theme.css{theme_v}">
{head_theme2}{head_katex}{extra_head}</head>
<body{body_cls}>
{nav_html}
{content}
{katex_script}<script src="{assets_rel}/ui.js{ui_v}"></script>
{xg2_script}{extra_body_end}</body>
</html>
"""


# ---------------------------------------------------------------- cross-page console nav
def nav_tabs(items, active):
    """A sticky cross-page tab bar for a project console — desktop underline-tabs
    that collapse into a horizontally-scrollable phone pill bar via one responsive
    CSS block (theme.css `.nav-tabs`), no separate mobile markup needed. Pass as
    `nav_html` to page(). Added 2026-07-10 (somages console v11).

    - items: list of (key, label, href) tuples, in display order.
    - active: the key of the current tab (gets the `.active` underline/pill state).
    """
    tabs = "".join(
        f'<a href="{_esc_attr(href)}" class="nav-tab{" active" if key == active else ""}">{_esc(label)}</a>'
        for key, label, href in items
    )
    return f'<nav class="nav-tabs"><div class="nav-tabs-inner">{tabs}</div></nav>'


def nav_subtabs(items, active):
    """A secondary row of smaller tabs under nav_tabs() — for a top-level tab that
    groups several pages (e.g. a console's Project docs family: plan/state/log/
    decisions/about). Same (key, label, href) / active contract as nav_tabs()."""
    tabs = "".join(
        f'<a href="{_esc_attr(href)}" class="nav-subtab{" active" if key == active else ""}">{_esc(label)}</a>'
        for key, label, href in items
    )
    return f'<nav class="nav-subtabs"><div class="nav-subtabs-inner">{tabs}</div></nav>'


def header(h1, sub_html=""):
    sub = f'<p class="sub">{sub_html}</p>' if sub_html else ""
    return f"""<header>
    <h1>{_esc(h1)}</h1>
    {sub}
  </header>"""


def _outline_nav(entries):
    def link(e, sub=False):
        cls = "ol-sublink" if sub else "ol-link"
        target = e.get("_target", e["id"])
        return (f'<a href="#{e["id"]}" class="{cls}" data-spy-link="{e["id"]}" '
                f'data-target="{target}">{_esc(e["label"])}</a>')

    items = []
    for e in entries:
        sub_html = ""
        if e.get("sub"):
            sub_items = []
            for s in e["sub"]:
                s = dict(s, _target=e["id"])
                sub_items.append(f'<li>{link(s, sub=True)}</li>')
            sub_html = f'\n      <ul class="outline-sub">{"".join(sub_items)}</ul>'
        e = dict(e, _target=e["id"])
        items.append(f'    <li>{link(e)}{sub_html}</li>')
    return f"""<nav class="outline" id="outline">
  <button class="expand-all-btn" id="expand-all-btn">Expand all</button>
  <ul class="outline-list">
{chr(10).join(items)}
  </ul>
</nav>"""


# ---------------------------------------------------------------- v3 workspace chrome
def v3_tree(entries, *, title="Somages", subtitle="", tree_src=None, switcher=None):
    """The left workspace page-tree nav for a theme="v3" page (2026-07-16).
    Same on every page of one workspace; pass the SAME entries to every page and
    mark exactly one leaf active so the tree reads as a persistent shell.

    entries: an ordered list of dicts. A LEAF is {"label", "href", "active"?,
    "meta"?} — "meta" is an optional small muted second line under the label
    (e.g. a daily report's subject: "experiment plan template").
    A GROUP is {"label", "children": [leaf, ...], "active"?} — its label is a
    small section heading (not a link), children are indented leaves (e.g.
    "Daily reports" over per-date entries). One level of nesting only.

    switcher (2026-07-16, console rollout): optional list of {"label","href"}
    rendered as a quiet zone-switch block between the tree head and the page
    list — how the operator console links INTO the research workspace.
    ZONE-BOUNDARY LAW (one-way rule, ratified 2026-07-16): only the console
    zone carries a switcher; nothing in the workspace zone links back to the
    console. Labels conventionally end with an arrow marker (↗).

    tree_src (2026-07-16, workspace rollout): a URL to the zone's tree.json
    ({"title","subtitle","entries":[...same schema...]}); emitted as a
    data-tree attribute that xg3.js fetches at LOAD, re-rendering the list
    and computing the active leaf from location.pathname — so every
    already-published page's sidebar tracks the zone's CURRENT tree (the
    baked entries are the no-JS fallback), and frozen /v/N/ snapshots stay
    content-immutable while their nav is live.

    Returns the <nav class="v3-tree"> markup. theme3.css styles it as a fixed
    left column at >=1200px and an off-canvas drawer (toggled by xg3.js) below."""
    def leaf(e):
        cls = "v3-tree-link" + (" active" if e.get("active") else "")
        meta = (f'<span class="v3-leaf-meta">{_esc(e["meta"])}</span>'
                if e.get("meta") else "")
        return (f'<li><a class="{cls}" href="{_esc_attr(e["href"])}">'
                f'{_esc(e["label"])}{meta}</a></li>')

    items = []
    for e in entries:
        if e.get("children"):
            kids = "".join(leaf(k) for k in e["children"])
            items.append(f'<li class="v3-tree-group"><div class="v3-tree-grouplabel">'
                         f'{_esc(e["label"])}</div><ul class="v3-tree-sublist">{kids}</ul></li>')
        else:
            items.append(leaf(e))
    sub = f'<div class="v3-brand-sub">{_esc(subtitle)}</div>' if subtitle else ""
    src_attr = f' data-tree="{_esc_attr(tree_src)}"' if tree_src else ""
    sw = ""
    if switcher:
        links = "".join(
            f'<a class="v3-zone-link" href="{_esc_attr(s["href"])}">{_esc(s["label"])}</a>'
            for s in switcher)
        sw = f'<div class="v3-zone-switch">{links}</div>'
    return (f'<nav class="v3-tree" id="v3-tree" aria-label="Workspace pages"{src_attr}>'
            f'<div class="v3-tree-head"><div class="v3-brand">{_esc(title)}</div>{sub}</div>'
            f'{sw}<ul class="v3-tree-list">{"".join(items)}</ul></nav>')


def _v3_outline_nav(entries):
    """The right per-page outline for a theme="v3" page: the current page's
    section list, with scrollspy highlight wired by xg3.js (keyed off each
    link's #id target, so no per-section markup change is needed). entries:
    list of {"id", "label"}. Rendered as a sticky right rail at >=1200px,
    hidden below (the tree drawer + in-flow content carry navigation there)."""
    if not entries:
        return ""
    lis = "".join(
        f'<li><a class="v3-ol-link" href="#{_esc_attr(e["id"])}" '
        f'data-spy-link="{_esc_attr(e["id"])}">{_esc(e["label"])}</a></li>'
        for e in entries)
    return (f'<aside class="v3-outline" id="v3-outline" aria-label="On this page">'
            f'<div class="v3-ol-title">On this page</div>'
            f'<ul class="v3-ol-list">{lis}</ul></aside>')


def v3_version_slot(*, version=None, date=None, note="", manifest="versions.json",
                    living=False):
    """The version-picker slot for a theme="v3" page. LIVING-PAGE-CANONICAL
    (user-ratified 2026-07-19, arXiv model): the stable URL always serves the
    LIVING page; published X.y versions are immutable /v/X.y/ bookmarks, not
    releases the stable page redirects to. Pass the result as page()'s
    version_slot=. SHELL UI + MANIFEST CONTRACT ONLY.

    Three modes:
    - LIVING page (living=True; version = the LATEST label): a quiet dropdown
      showing just "Version X.y" (no date -- a page edited today must not read
      as stale via an old label's date). xg3.js fills the menu from
      ./versions.json (a list of {"v": "X.y" string, "date", "note",
      "snapshot": bool} newest-last) with a "Current version" row (the living
      page, marked here) above the labeled snapshots. NO not-current banner.
    - SNAPSHOT page (living=False; version = the snapshot's own label): shows
      "Version X.y · <date>" and, at runtime, the "Version X.y, not current,
      view current version" banner linking back to the living page (manifest
      path is ../../versions.json).
    - DATE-FROZEN page (daily reports; version is None): plain mono date text,
      no picker (the date IS the version)."""
    if version is None:
        return f'<div class="v3-vslot"><span class="v3-vdate">{_esc(date)}</span></div>\n'
    if living:
        label = f'Version {_esc(str(version))}'
        return (f'<details class="v3-vslot v3-vpick" data-versions="{_esc_attr(manifest)}" '
                f'data-current="" data-living="1">'
                f'<summary>{label}</summary>'
                f'<ul class="v3-vmenu"><li class="v3-verr">loading history&hellip;</li></ul>'
                f'</details>\n')
    note_html = f' <span class="v3-vnote">{_esc(note)}</span>' if note else ""
    label = f'Version {_esc(str(version))} &middot; {_esc(date)}'
    return (f'<details class="v3-vslot v3-vpick" data-versions="{_esc_attr(manifest)}" '
            f'data-current="{_esc_attr(str(version))}">'
            f'<summary>{label}{note_html}</summary>'
            f'<ul class="v3-vmenu"><li class="v3-verr">loading history&hellip;</li></ul>'
            f'</details>\n')


def _v3_topbar(nav_title=""):
    """The mobile top bar for a theme="v3" page — a sticky strip carrying the
    tree-toggle (hamburger) button and a short page title. Shown only below the
    3-column breakpoint (theme3.css); the button toggles the tree drawer via
    xg3.js. Always emitted; CSS hides it at wide widths."""
    t = f'<span class="v3-topbar-title">{_esc(nav_title)}</span>' if nav_title else ""
    return (f'<div class="v3-topbar"><button class="v3-menu" id="v3-menu" type="button" '
            f'aria-label="Toggle navigation" aria-expanded="false">'
            f'<span class="v3-menu-bars"></span></button>{t}</div>')


def _katex_runtime_script(assets_rel):
    return f'''<script src="{assets_rel}/katex/katex.min.js"></script>
<script>
document.querySelectorAll('.kx').forEach(function(el) {{
  try {{
    katex.render(el.getAttribute('data-tex'), el, {{
      displayMode: el.getAttribute('data-display') === '1',
      throwOnError: false
    }});
  }} catch (e) {{
    el.textContent = el.getAttribute('data-tex');
    console.error('KaTeX render failed', e);
  }}
}});
</script>
'''


# ---------------------------------------------------------------- sections
def section(id, num, title, *, takeaway=None, body_html="", preview_rem=None, spy=True):
    """One <section>. `num` is shown as a leading badge (e.g. 2 -> "2 Results: ...");
    pass None/"" to omit it. If preview_rem is given, the section is collapsible
    (Medium-style preview/expand, cut at preview_rem rem of height); if None, the
    section is always fully visible (e.g. Overview) with no fade/button chrome.
    `spy` adds data-spy so the outline's scrollspy can track this section.
    """
    num_html = f'<span class="num">{_esc(str(num))}</span>' if num not in (None, "") else ""
    heading = f'<h2 id="{id}-h">{num_html}{_esc(title)}</h2>' if False else f'<h2>{num_html}{_esc(title)}</h2>'
    takeaway_html = f'<p class="takeaway">{takeaway}</p>' if takeaway else ""
    spy_attr = " data-spy" if spy else ""

    if preview_rem is None:
        return f'''  <section id="{id}"{spy_attr}>
    {heading}
    {takeaway_html}
    {body_html}
  </section>'''

    return f'''  <section id="{id}" class="collapsible" style="--preview-h: {preview_rem}rem"{spy_attr}>
    {heading}
    <div class="section-body">
    {takeaway_html}
    {body_html}
    </div>
    <div class="fade-overlay"><button class="expand-btn">Expand section &#9662;</button></div>
  </section>'''


def callout(inner_html, warn=False, title=None):
    """title (added 2026-07-14, v2): optional bold lead line rendered as its own
    <p class="t"> above the body — the v2 pattern where a callout opens with a
    2-4 word claim ("The pilot number was pool-enriched.") instead of an inline
    <b>Verdict.</b> prefix. Omit (default) for the original rendering, which is
    byte-identical to pre-v2 builds."""
    style = ' style="border-left-color:var(--accent2)"' if warn else ""
    title_html = f'<p class="t">{title}</p>' if title else ""
    return f'<div class="callout"{style}>{title_html}{inner_html}</div>'


def honesty_box(inner_html):
    return f'<div class="honesty-box">&#9888; {inner_html}</div>'


def pill(text, cls=""):
    return f'<span class="tag {cls}">{_esc(text)}</span>'


def filepath(basename, full_path, kind=""):
    """A file mention that reveals its full path on hover and copies it on click
    (see web/assets/ui.js's code.fpath click handler). Use ONLY for cluster/local
    paths that aren't web-linkable — a GitHub blob, arXiv abstract, or another
    published page should stay a plain <a>, not this component.
    - basename: what's shown (e.g. "data_splits_74k.json")
    - full_path: the real path, verified against the cluster — never guessed;
      shown as the hover title and what gets copied.
    - kind: optional trailing annotation already used elsewhere on the page
      (e.g. "(cluster)") — appended as plain text after the element, not part
      of the copyable path, so callers keep their existing access-marker
      convention (see the funnel's public/private/cluster legend) alongside it.
    """
    kind_html = f' {kind}' if kind else ""
    return (f'<code class="fpath" data-path="{_esc_attr(full_path)}" '
            f'title="{_esc_attr(full_path)}">{_esc(basename)}</code>{kind_html}')


# ---------------------------------------------------------------- overview components
def verdict_box(inner_html):
    return f'<div class="verdict">{inner_html}</div>'


def hero_figs(cards_html):
    """cards_html: list of already-built .hero-fig card HTML strings (see hero_fig_row /
    hero_fig_image). Wraps them in the responsive .hero-figs flex row."""
    return f'<div class="hero-figs">\n{chr(10).join(cards_html)}\n</div>'


def hero_fig_row(title, cells):
    """One .hero-fig card containing a row of equal-size image cells.
    cells: list of (img_src, caption) tuples."""
    cell_html = "".join(
        f'<div class="hrcell"><img src="{_esc(src)}"><div class="hrcap">{_esc(cap)}</div></div>'
        for src, cap in cells
    )
    return f'''<div class="hero-fig">
        <div class="hf-title">{_esc(title)}</div>
        <div class="hero-row">{cell_html}</div>
      </div>'''


def hero_fig_image(title, img_src, alt=""):
    """One .hero-fig card containing a single full-width image (e.g. a captured chart)."""
    return f'''<div class="hero-fig">
        <div class="hf-title">{_esc(title)}</div>
        <img src="{_esc(img_src)}" alt="{_esc(alt)}">
      </div>'''


def next_bullets(items):
    """items: list of raw HTML strings (may contain <strong> etc.)."""
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<ol class="next-bullets">{lis}</ol>'


def stat_tiles(tiles, cls=""):
    """tiles: list of (value, label) pairs -> a .stat-row of .stat tiles.
    cls: optional modifier class per call (e.g. "train" / "val") applied to every tile
    in this row; call twice (once per split) for a train/val pair of rows."""
    cell_cls = f"stat {cls}".strip()
    cells = "".join(f'<div class="{cell_cls}"><b>{_esc(str(v))}</b><span>{_esc(l)}</span></div>'
                     for v, l in tiles)
    return f'<div class="stat-row">{cells}</div>'


# ---------------------------------------------------------------- tables
def results_table(headers, rows_html, first_col_left=True):
    """headers: list of column label strings. rows_html: pre-built <tr>...</tr> strings
    (row internals vary too much per page to generalize — build rows with your own
    f-strings using the table.results / .rowhead / .rawinput classes from theme.css,
    then pass the joined string here)."""
    th = "".join(
        f'<th style="text-align:left">{_esc(h)}</th>' if i == 0 and first_col_left else f'<th>{_esc(h)}</th>'
        for i, h in enumerate(headers)
    )
    return f'''<div class="table-scroll">
    <table class="results">
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}
      </tbody>
    </table>
    </div>'''


def tensor_table(rows, id=None, extra_html=""):
    """rows: list of dicts {name, shape, role, cls (optional: "target" or "aux")}.
    extra_html: raw HTML appended inside the same .tensor-wrap card (e.g. an
    equation block or diagram that belongs visually with the tensor inventory)."""
    trs = []
    for r in rows:
        cls = f' class="{r["cls"]}"' if r.get("cls") else ""
        trs.append(f'<tr{cls}><td class="tname">{_esc(r["name"])}</td>'
                   f'<td class="tshape">{_esc(r["shape"])}</td><td>{r["role"]}</td></tr>')
    id_attr = f' id="{id}"' if id else ""
    return f'''<div class="tensor-wrap"{id_attr}>
      <div class="table-scroll">
      <table class="tensors">
        <thead><tr><th>tensor</th><th>shape</th><th>role</th></tr></thead>
        <tbody>
          {"".join(trs)}
        </tbody>
      </table>
      </div>
      {extra_html}
    </div>'''


def legend(items):
    """items: list of raw HTML strings, each one legend entry."""
    return f'<div class="legend">{"".join(f"<span>{i}</span>" for i in items)}</div>'


def legend_swatch(color, label):
    """One colored-square + label entry, for feeding into legend() when the
    distinguishing feature is a color (e.g. a per-class legend) rather than the
    line-style swatch legend()'s own <span class="sw"> already covers."""
    return f'<span class="lg-sw" style="background:{_esc_attr(color)}"></span>{_esc(label)}'


# ---------------------------------------------------------------- badges
def badge(text, bg=None, color=None):
    """One small solid-fill chip (as opposed to pill()'s outlined tag) — for dense
    per-row metadata like per-class scores or counts. bg/color: optional inline
    color overrides (hex or CSS color name); omit both for the neutral default."""
    style = ""
    if bg:
        style += f"background:{_esc_attr(bg)};"
    if color:
        style += f"color:{_esc_attr(color)};"
    style_attr = f' style="{style}"' if style else ""
    return f'<span class="badge"{style_attr}>{_esc(text)}</span>'


# ---------------------------------------------------------------- collapsible About blocks
def details_block(summary_html, body_html, open=False):
    """A native <details>/<summary> disclosure widget — for short reference asides
    inside a section (About/purpose, "what do these classes mean", "how is this
    computed"), as distinct from section()'s Medium-style preview/expand (which
    trades off an entire section's EVIDENCE, not a small aside). Zero-JS: relies on
    the browser's native <details> toggle, so it works even on pages with ui.js
    absent. Stack several under one Overview/Method section for a multi-aside
    "About this page" cluster."""
    open_attr = " open" if open else ""
    return f'''<details class="xdetails"{open_attr}>
    <summary>{summary_html}</summary>
    <div class="xdetails-body">{body_html}</div>
  </details>'''


# ---------------------------------------------------------------- pseudo-code / annotated blocks
def code_block(inner_html, cls="pseudo"):
    """A monospace block for pseudo-code or annotated code, pre-tokenized by the
    caller with span classes: .kw (keyword), .cm (comment), .num (number/constant)
    are pre-styled by theme.css; for domain-specific colored tokens (e.g. per-class
    names in a classifier's pseudo-code), use inline style="color:..." spans or add
    project-specific classes to your own page's <style> — code_block does not
    invent a token taxonomy, it only provides the block chrome + the three generic
    token classes above. inner_html is inserted verbatim (already-escaped/tokenized
    by the caller, matching every other xgpage component's "you bring HTML" contract)."""
    return f'<pre class="{_esc_attr(cls)}">{inner_html}</pre>'


def code_token(text, cls=None, color=None):
    """One tokenized span for use inside code_block(). cls: one of the generic
    theme.css token classes ("kw"/"cm"/"num"); color: an inline color override for
    tokens that need a page-specific palette (e.g. 5 class-name colors) that
    code_block's 3 generic classes don't cover."""
    if color:
        return f'<span style="color:{_esc_attr(color)}">{_esc(text)}</span>'
    cls_attr = f' class="{_esc_attr(cls)}"' if cls else ""
    return f'<span{cls_attr}>{_esc(text)}</span>'


# ---------------------------------------------------------------- wide single-image row gallery
def wide_row(id, header_html, img_src, alt="", spy=False):
    """One full-width list row: a compact header line (name + badges) over a single
    (often multi-panel-composited) image, horizontally scrollable if wider than the
    viewport. Distinct from gallery_card/gallery_grid (which pair exactly two
    images per card in a grid) — use this for N rows of one image each, e.g. a
    per-shape montage list sorted by some score. Rows are plain <div>s, not
    <section>s, so hundreds can be emitted cheaply without per-row scrollspy cost;
    pass spy=True only if the page needs to jump-link individual rows from the
    outline (rare at this row count)."""
    spy_attr = " data-spy" if spy else ""
    id_attr = f' id="{_esc_attr(id)}"' if id else ""
    return f'''<div class="wide-row"{id_attr}{spy_attr}>
    <div class="wide-row-head">{header_html}</div>
    <div class="wide-row-imgwrap"><img loading="lazy" src="{_esc_attr(img_src)}" alt="{_esc_attr(alt)}"></div>
  </div>'''


def wide_gallery(rows_html):
    """Wraps wide_row() strings in the .wide-gallery list container."""
    return f'<div class="wide-gallery">{"".join(rows_html)}</div>'


# ---------------------------------------------------------------- equations (KaTeX)
def equation(tex, *, display=True, comment=None, tag=None):
    """One .eq-row: a KaTeX span (rendered client-side by the runtime script from
    page(needs_katex=True)) plus an optional inline comment and/or a leading tag pill
    (e.g. <span class="tag w5">W5</span>-style badge text)."""
    tag_html = f'<span class="eq-tag">{_esc(tag)}</span>' if tag else ""
    disp = "1" if display else "0"
    comment_html = f'<div class="eqcomment">{comment}</div>' if comment else ""
    return (f'<div class="eq-row">{tag_html}<div class="eqbox">'
            f'<span class="kx" data-display="{disp}" data-tex="{_esc_attr(tex)}"></span></div>'
            f'{comment_html}</div>')


def inline_katex(tex):
    """An inline (non-display) KaTeX span for use inside a normal <p>, e.g.
    f'... noise {lp.inline_katex(r"\\varepsilon \\sim \\mathcal{{N}}(0,I)")} ...'."""
    return f'<span class="kx" data-tex="{_esc_attr(tex)}"></span>'


# ---------------------------------------------------------------- provenance / funnel
def flow_stage(num, label, sub="", highlight=False, extra_cls=""):
    cls = ("flow-stage highlight" if highlight else "flow-stage")
    if extra_cls:
        cls += f" {extra_cls}"
    subhtml = f'<div class="fsub">{sub}</div>' if sub else ""
    return f'<div class="{cls}"><div class="fnum">{_esc(str(num))}</div><div class="flbl">{label}</div>{subhtml}</div>'


def flow_arrow(label=""):
    return f'<div class="flow-arrow"><span class="arrowglyph">&darr;</span>{label}</div>'


def flow_branch(stages_html):
    return f'<div class="flow-branch">{"".join(stages_html)}</div>'


def flow_wrap(inner_html):
    return f'<div class="flow-wrap"><div class="flow">{inner_html}</div></div>'


# ---------------------------------------------------------------- gallery grid
def gallery_card(img_left, img_right, cap_left, cap_right):
    return f'''<div class="card">
          <div class="pair"><img src="{_esc(img_left)}"><img src="{_esc(img_right)}"></div>
          <div class="cardcap"><span class="csid">{cap_left}</span><span class="cfrac">{cap_right}</span></div>
        </div>'''


def gallery_grid(cards_html):
    return f'<div class="cardgrid">{"".join(cards_html)}</div>'


# ---------------------------------------------------------------- labeled comparison grid
def grid_figure(row_labels, col_labels, cells, *, id=None, export_link=None, caption=None):
    """A labeled comparison grid (rows x columns of images with real-text headers) —
    replaces PIL-baked composite comparison PNGs (2026-07-07 convention: real labels
    are selectable/searchable, cells link to full-res panels, missing runs are an
    explicit placeholder instead of a blank tile). Not a <table> (images need
    object-fit control table cells don't give cleanly) — a CSS grid.

    - row_labels: list of strings, one per row (e.g. per-shape ids).
    - col_labels: list of strings, one per column (e.g. "Input" / "GT" / "Method A").
    - cells: list of rows matching row_labels, each row a list of dicts matching
      col_labels: {"img": src, "href": optional (defaults to img itself), "caption":
      optional short HTML under the image (e.g. a stat line — wrap significant
      figures in <span class="gf-sig"> for the red-highlight convention), "alt":
      optional, "placeholder": optional text — renders a dashed-border text tile
      instead of an <img>, for "not run"/missing cells}.
    - export_link: optional {"href", "label"} — a small link above the grid to the
      archival composite PNG (e.g. "Download composite figure (PNG)"), kept as an
      export artifact, not the page's primary presentation.
    - caption: optional overall figure caption below the grid.

    Horizontally scrollable in its OWN .grid-figure-scroll wrapper on narrow
    viewports (min-width floor on .grid-figure) — page-level scrollX must stay 0;
    see xgpage SKILL.md rules 5/10 (wide content scrolls inside its own wrapper,
    and a flex ancestor needs min-width:0 to let that wrapper's overflow:auto work)."""
    id_attr = f' id="{_esc_attr(id)}"' if id else ""
    export_html = ""
    if export_link:
        export_html = (f'<div class="grid-figure-export"><a href="{_esc_attr(export_link["href"])}">'
                        f'{_esc(export_link.get("label", "Download composite figure (PNG)"))}</a></div>')
    header_cells = '<div class="gf-cell gf-corner"></div>' + "".join(
        f'<div class="gf-cell gf-colhead">{_esc(c)}</div>' for c in col_labels)
    body_rows = []
    for rlabel, row in zip(row_labels, cells):
        row_html = [f'<div class="gf-cell gf-rowhead">{_esc(rlabel)}</div>']
        for c in row:
            if c.get("placeholder"):
                row_html.append(f'<div class="gf-cell gf-imgcell gf-placeholder">{_esc(c["placeholder"])}</div>')
            else:
                cap = f'<div class="gf-cap">{c["caption"]}</div>' if c.get("caption") else ""
                href = c.get("href", c["img"])
                row_html.append(
                    f'<div class="gf-cell gf-imgcell"><a href="{_esc_attr(href)}">'
                    f'<img loading="lazy" src="{_esc_attr(c["img"])}" alt="{_esc_attr(c.get("alt", ""))}">'
                    f'</a>{cap}</div>')
        body_rows.append("".join(row_html))
    grid_html = (f'<div class="grid-figure"{id_attr} '
                 f'style="grid-template-columns: auto repeat({len(col_labels)}, 1fr);">'
                 f'{header_cells}{"".join(body_rows)}</div>')
    caption_html = f'<div class="gf-caption">{caption}</div>' if caption else ""
    return f'{export_html}<div class="grid-figure-scroll">{grid_html}</div>{caption_html}'


def bucket_block(title_html, cards_html):
    return f'<div class="bucket-block"><h3>{title_html}</h3>{gallery_grid(cards_html)}</div>'


def thumb_card(img_src, title, tags=None, cls=""):
    """A single-thumbnail card (one image, a title line, up to a few tag pills) —
    for galleries where each item is one object, not an appearance/target pair
    (that's gallery_card). Shares .card/.cardgrid so it drops into gallery_grid()
    interchangeably with gallery_card()."""
    tags_html = ""
    if tags:
        tags_html = '<div class="thumbtags">' + "".join(f'<span class="thumbtag">{_esc(t)}</span>' for t in tags) + '</div>'
    cls_attr = f" {cls}" if cls else ""
    return f'''<div class="card thumb-card{cls_attr}">
          <img class="thumb-card-img" src="{_esc(img_src)}">
          <div class="thumb-card-title">{_esc(title)}</div>
          {tags_html}
        </div>'''


# ---------------------------------------------------------------- misc
def chart_wrap(svg_or_img_html):
    return f'<div class="chart-wrap">{svg_or_img_html}</div>'


def _esc(s):
    # NOT an HTML-escape: every caller here is first-party content (this project's own
    # authored copy/sids), which routinely already contains intentional HTML (entities
    # like &mdash;, inline <code>/<em> tags). Escaping it would double-escape those
    # entities. Kept as a named passthrough (rather than removing the calls) so the
    # call sites stay self-documenting about "this is a text-ish field."
    return str(s)


def _esc_attr(s):
    return _html.escape(str(s), quote=True)


# ================================================================ v2 components
# Design-language v2 (2026-07-14, xgpage-designer commission): editorial pages
# in the Anthropic palette where every important claim is carried by a visual.
# ALL functions below require page(theme="v2") — they emit markup styled only by
# assets/theme2.css. Everything above is untouched v1 and keeps rendering
# byte-identically on existing pages. Composition rhythm per section:
#   kicker (mono "01") -> claim-as-heading (h2 states the FINDING) ->
#   visual (figure / chart / compare) -> short prose (inside prose()).
# See the xgpage SKILL.md design principles for when to use what.


def statband(stats):
    """The headline numbers, once, under the hero dek — a bordered flex band of
    serif numerals with small uppercase labels (NOT boxes; distinct from v1
    stat_tiles). stats: list of (value, label) pairs, 3-6 of them. The value is
    a short string ("45,374", "17.5%", "66 lines"); the label a few words."""
    items = "".join(
        f'<div class="sb-item"><div class="sb-n">{_esc(v)}</div>'
        f'<div class="sb-l">{_esc(l)}</div></div>'
        for v, l in stats)
    return f'<div class="statband">{items}</div>'


def kicker(text):
    """The small mono uppercase line above a section h2 (usually a 2-digit
    section number like "01", optionally "01 · corpus scale")."""
    return f'<div class="kicker">{_esc(text)}</div>'


def toc_pills(entries):
    """A flat, wrapping table of contents under the hero: numbered inline links.
    entries: list of (id, label); numbers are derived (01, 02, ...)."""
    lis = "".join(
        f'<li><a href="#{_esc_attr(id_)}"><span class="num">{i + 1:02d}</span>'
        f'{_esc(label)}</a></li>'
        for i, (id_, label) in enumerate(entries))
    return f'<nav class="toc" aria-label="Contents"><ol>{lis}</ol></nav>'


def hero_header(eyebrow_text, title, dek_html="", stats=None, toc=None):
    """The v2 page opening: mono eyebrow, big serif h1, dek paragraph, optional
    statband + toc pills. Replaces header() on v2 pages.
    - eyebrow_text: "project · page kind · provenance claim" mono line.
    - dek_html: 1-3 sentence standfirst (raw HTML allowed).
    - stats: statband() input, or None.
    - toc: toc_pills() input, or None."""
    dek = f'<p class="dek">{dek_html}</p>' if dek_html else ""
    band = statband(stats) if stats else ""
    toc_html = toc_pills(toc) if toc else ""
    return (f'<header><div class="eyebrow">{_esc(eyebrow_text)}</div>'
            f'<h1>{_esc(title)}</h1>{dek}{band}{toc_html}</header>')


def section_v2(id, num, title, body_html, *, spy=False):
    """A v2 section: kicker number above a claim-stated h2 (the heading is the
    finding — "Contact scales with chart count", not "Results"). No preview/
    collapse mechanic: v2 pages curate instead of folding. spy=True only if the
    page also mounts the v1 outline sidebar (rare; v2 pages use toc_pills)."""
    spy_attr = " data-spy" if spy else ""
    num_html = kicker(f"{num:02d}" if isinstance(num, int) else str(num)) if num not in (None, "") else ""
    return (f'<section id="{_esc_attr(id)}"{spy_attr}>{num_html}'
            f'<h2>{_esc(title)}</h2>\n{body_html}\n</section>')


def prose(inner_html):
    """Running text on a v2 page lives inside this 680px-measure wrapper —
    never as bare <p> against the full 1020px column."""
    return f'<div class="prose">{inner_html}</div>'


def _fig_size(native_px, content):
    """Intrinsic-size-aware figure sizing (design law D9, user-set 2026-07-15:
    'determine the image size by its resolution / its content' — a 512-pixel
    UV map blown past 680px is blocky pixels spending a full column of
    vertical space on low information density).
    Returns (display_cap_px_or_None, pixelated_bool) for one image:
    - content='pixel-map' (idmaps, UV canvases, clsmaps, any blocky raster):
      never ABOVE native size, and big maps default smaller (420px cap for
      >=512px sources); rendered with image-rendering:pixelated so what does
      show stays crisp.
    - content='photo' (renders, photographs): never upscale beyond 1.25x
      native — an undersized source shows small, it does not smear.
    - content='chart'/None: no cap (vector SVGs and purpose-sized exports).
    Callers should also CROP sparse sources to their content bbox before
    sizing — an almost-empty canvas earns even fewer pixels."""
    if not native_px:
        return (None, content == "pixel-map")
    if content == "pixel-map":
        return (min(int(native_px), 420), True)
    if content == "photo":
        return (round(native_px * 1.25), False)
    return (None, False)


def _center_frag(width_px):
    """Centered-breakout style fragment (design law D11, user-set 2026-07-15,
    following distill.pub; corrected 2026-07-15 same day): any figure that
    carries an explicit max-width centers with auto margins, UNCONDITIONALLY
    — no pixel threshold. This is deliberately container-relative rather
    than absolute-px: CSS resolves `margin:auto` to 0 (i.e. flush, reads as
    "left") the instant the figure's max-width equals its containing
    block's width (the prose() 680px case, or a matrix that exactly fills
    the 972px page column), and only produces visible centering when the
    figure is genuinely narrower than whatever it happens to sit inside.
    An earlier version gated this on `width_px > 680`, which silently broke
    for any breakout-class figure (method_matrix, fig_row, ...) placed
    directly under a section_v2() body — 972px wide, NOT wrapped in
    prose() — whose own computed width landed under 680px but still well
    under its real 972px container: that figure hugged the section's left
    edge with the shortfall dumped as dead space on the right. Caught live
    on the daily-report aesthetic-rating matrix (672px wide inside a 972px
    section): a steady 300px one-sided overhang at every viewport width,
    the same defect class D11 was written to kill, just under the
    threshold's radar. Returns the inline-style tail to append after a
    max-width declaration."""
    return ";margin-left:auto;margin-right:auto" if width_px else ""


def fig(img_src, caption_html="", alt="", loading_lazy=True,
        native_px=None, content=None):
    """One figure with a figcaption that ARGUES: open the caption with a bolded
    finding ("<b>The grid is real.</b> ..."), not a restatement of what the
    image is. caption_html is raw HTML.
    native_px/content: intrinsic-size hints (see _fig_size / SKILL.md D9) —
    pass the source's pixel width and 'pixel-map'|'photo'|'chart' so the
    display size follows the content instead of the column."""
    cap_px, pixelated = _fig_size(native_px, content)
    style = f' style="max-width:{cap_px}px{_center_frag(cap_px)}"' if cap_px else ""
    px_cls = ' class="fig-px"' if pixelated else ""
    lazy = ' loading="lazy"' if loading_lazy else ""
    cap = f'<figcaption>{caption_html}</figcaption>' if caption_html else ""
    return (f'<figure{style}><img{lazy}{px_cls} src="{_esc_attr(img_src)}" '
            f'alt="{_esc_attr(alt)}">{cap}</figure>')


def fig_row(panels, caption_html="", native_px=None, content=None):
    """2-3 labeled image panels side by side (stack on phones), one shared
    argued caption below. panels: list of (panel_label, img_src) or
    (panel_label, img_src, alt); panel labels are the small mono uppercase
    line ABOVE each image ("the scene · 2,230,190 tris").
    native_px/content: intrinsic-size hints applied to EVERY panel (see
    _fig_size / D9) — the row's total width is capped at n·cap+gaps so
    same-content panels never blow up past their information density."""
    cls = "fig-triple" if len(panels) == 3 else "fig-pair"
    cap_px, pixelated = _fig_size(native_px, content)
    gap = 14 if len(panels) == 3 else 18
    row_w = cap_px * len(panels) + gap * (len(panels) - 1) if cap_px else None
    row_style = (f' style="max-width:{row_w}px{_center_frag(row_w)}"'
                 if cap_px else "")
    px_cls = ' class="fig-px"' if pixelated else ""
    cells = []
    for p in panels:
        label, src, alt = (tuple(p) + ("",))[:3]
        cells.append(
            f'<div><div class="panel-label">{label}</div>'
            f'<figure style="margin:0"><img loading="lazy"{px_cls} '
            f'src="{_esc_attr(src)}" alt="{_esc_attr(alt)}"></figure></div>')
    cap = (f'<figure style="margin-top:14px"><figcaption>{caption_html}'
           f'</figcaption></figure>') if caption_html else ""
    return f'<div class="{cls}"{row_style}>{"".join(cells)}</div>{cap}'


def method_matrix(columns, rows, caption_html="", native_px=None, content=None, id=None,
                  page_inner=972):
    """The benchmark qualitative-comparison grid (2026-07-15, from the daily-report
    strip retrofit): columns = METHODS, rows = CASES — the way a SIGGRAPH paper
    figure sets it. THE pattern for methods-x-cases qualitative sections; don't
    hand-roll flex strips per row.

    Grid contract (what makes it read as one figure, not stacked strips):
    - Column headers appear ONCE at the top — mono kicker style, uppercase,
      nowrap. Wrapping is structurally impossible: labels are length-CHECKED
      against the computed cell width and a too-long label raises ValueError.
      Labels NAME the method; qualifiers ("(TRELLIS-2)", "post-hoc") move to
      caption_html. Same rule for row labels (rotated, left gutter,
      typographically distinct from the mono column headers).
    - Every cell is an identical square tile: object-fit:contain on the same
      light background all v2 figures use, same 1px border + 8px radius.
    - A missing/pending cell is {"placeholder": label, "sub": optional} — it
      renders IN-SYSTEM as a tinted, subtly hatched cell with a small centered
      mono label, not a special box (and not a baked PNG).
    - ONE argued figcaption under the whole matrix (D4): open with the bolded
      finding, then the column/row reading instructions and the qualifiers.
      The caption's width is LOCKED to the rendered matrix width (inline
      max-width) so figure+caption read as one object — a SINGLE text
      column spanning the figure, the ACM TOG full-width-figure convention
      (long measure is acceptable in caption register; if it bothers you,
      tighten the text or drop the caption type size, never columns —
      a 2-col flow shipped 2026-07-15 and was rejected same day).
    - Centered breakout (2026-07-15, user-set, following distill.pub;
      corrected same day, fourth round): the matrix ALWAYS centers itself
      in the page column (auto margins via `_center_frag`, unconditional —
      no width threshold), with symmetric overhang whenever the tile span
      is narrower than its actual container. The caption is width-locked to
      the tile span, so it centers with the figure. The rotated row-label
      gutter still hangs OUTSIDE the figure box (-32px pull on .mm-scroll at
      >=1060px viewports), left of the centered tiles: the tiles are the
      visual mass being centered, and the hang keeps the gutter from eating
      tile width. Below 1060px the gutter sits inside the scroll box
      (labels scroll with the grid, no page overflow). (Supersedes two
      earlier attempts same day: first a left-anchored breakout with the
      first tile edge on the prose left edge, then a `width_px > 680`
      threshold that missed sub-680px matrices placed directly in a 972px
      section — see `_center_frag`'s docstring for the live repro.)
    - Phones/narrow: the grid keeps its computed px tracks and scrolls INSIDE
      its own .mm-scroll wrapper (SKILL.md rule 5) — zero page-level overflow.

    - columns: list of short label strings (rendered uppercase).
    - rows: list of (row_label, cells); each cells list matches columns and
      holds either an img src string, an image dict, or a placeholder dict.
      Image dict: {"img": src (alias "src"), "alt":, "badge":, "best":}.
      "badge" is a short per-cell score/label pinned to the cell's
      bottom-right corner (mono, translucent dark backing so it reads over
      white renders in both themes); "best": True renders that badge in the
      accent so the row's winner reads at a glance (one per row, D7's one
      accent). Numbers that belong to individual cells go in badges, not in
      the caption; the caption explains what the badge numbers ARE. Tracks
      are fixed px at every viewport (phones scroll), so badges keep their
      size and stay legible at any width.
    - native_px/content: D9 intrinsic-size hints for the cell images
      ('pixel-map' also renders pixelated); cell size is
      min(tile-span fit to the page_inner column, 220, D9 cap), floored at
      110px (below that the wrapper scrolls on desktop too).
    - page_inner: the page column's inner width the tile math fits
      (2026-07-16, v3): default 972 = the v2 page column (1020 max-width
      minus 2x24 padding), byte-identical to before the param existed.
      theme="v3" pages pass 820 (the v3 content measure) so a wide matrix
      lays tiles for the narrower column instead of scrolling inside
      .mm-scroll on desktop (SKILL rule 13's defect class).
    """
    n = len(columns)
    if n == 0:
        raise ValueError("method_matrix needs at least one column")
    for rlabel, cells in rows:
        if len(cells) != n:
            raise ValueError(f"row {rlabel!r} has {len(cells)} cells for {n} columns")
    cap, pixelated = _fig_size(native_px, content)
    gap, gutter = 6, 26
    # the row-label gutter hangs OUTSIDE the page column (theme2 pulls
    # .mm-scroll left by gutter+gap=32px — keep in sync with the CSS), so the
    # TILES get the full column: n cells + (n-1) gaps fit page_inner exactly
    cell = max(min((page_inner - gap * (n - 1)) // n, 220), 110)
    if cap:
        cell = min(cell, cap)
    tile_span = n * cell + (n - 1) * gap
    # mono col headers ≈7.9px/char at .68rem+.08em tracking; sans row labels
    # ≈6.6px/char at .74rem. A label wider than its track would collide with
    # its neighbor — fail loudly at build time instead.
    col_budget = int(cell / 7.9)
    row_budget = int(cell / 6.6)
    for c in columns:
        if len(c) > col_budget:
            raise ValueError(
                f"column label {c!r} is {len(c)} chars; max {col_budget} for "
                f"{cell}px cells — shorten it and move qualifiers into "
                f"caption_html (labels name, captions qualify)")
    px_cls = ' class="fig-px"' if pixelated else ""
    parts = ['<div class="mm-gutter"></div>']
    parts += [f'<div class="mm-col">{_esc(c)}</div>' for c in columns]
    for rlabel, cells in rows:
        if len(rlabel) > row_budget:
            raise ValueError(
                f"row label {rlabel!r} is {len(rlabel)} chars; max {row_budget} "
                f"for {cell}px cells — shorten it; qualifiers go in caption_html")
        parts.append(f'<div class="mm-rowlab">{_esc(rlabel)}</div>')
        for c in cells:
            if isinstance(c, str):
                c = {"src": c}
            if c.get("placeholder"):
                sub = f'<span class="mm-ph-sub">{_esc(c["sub"])}</span>' if c.get("sub") else ""
                parts.append(f'<div class="mm-cell mm-ph"><span>{_esc(c["placeholder"])}</span>{sub}</div>')
            else:
                src = c.get("img", c.get("src"))
                badge = ""
                if c.get("badge") is not None:
                    bcls = "mm-badge best" if c.get("best") else "mm-badge"
                    badge = f'<span class="{bcls}">{_esc(c["badge"])}</span>'
                parts.append(f'<div class="mm-cell"><img loading="lazy"{px_cls} '
                             f'src="{_esc_attr(src)}" alt="{_esc_attr(c.get("alt", ""))}">{badge}</div>')
    id_attr = f' id="{_esc_attr(id)}"' if id else ""
    fig_style = f' style="max-width:{tile_span}px{_center_frag(tile_span)}"'
    cap_html = ""
    if caption_html:
        cap_html = (f'<figcaption style="max-width:{tile_span}px">'
                    f'{caption_html}</figcaption>')
    return (f'<figure class="mm"{id_attr}{fig_style}><div class="mm-scroll">'
            f'<div class="method-matrix" style="grid-template-columns:'
            f'{gutter}px repeat({n},{cell}px)">{"".join(parts)}</div></div>'
            f'{cap_html}</figure>')


def compare_slider(img_left, img_right, label_left, label_right,
                   alt_left="", alt_right="", caption_html="",
                   native_px=None, content=None):
    """Two same-size, pixel-aligned images with a draggable divider: img_left
    is revealed LEFT of the divider (tag pill top-left), img_right is the base
    layer (tag pill top-right). Requires page(theme="v2") for xg2.js wiring.
    Degrades to the 50/50 split with no JS.

    RESERVED component (design law D10, user-set 2026-07-15): the DEFAULT
    comparison presentation is a static side-by-side (fig_row) or aligned
    small multiples — both states in one glance, zero interaction cost. Reach
    for the slider only when precise pixel-registration inspection IS the
    finding (same-view before/after where sub-panel misalignment matters),
    and consider pairing it with a static side-by-side even then.
    native_px/content: intrinsic-size hints (see _fig_size / D9)."""
    cap_px, pixelated = _fig_size(native_px, content)
    style = f' style="max-width:{cap_px}px{_center_frag(cap_px)}"' if cap_px else ""
    px_cls = ' class="fig-px"' if pixelated else ""
    cap = f'<figure style="margin-top:0"><figcaption>{caption_html}</figcaption></figure>' if caption_html else ""
    return f'''<div class="compare"{style}>
    <img{px_cls} src="{_esc_attr(img_right)}" alt="{_esc_attr(alt_right)}">
    <div class="top"><img{px_cls} src="{_esc_attr(img_left)}" alt="{_esc_attr(alt_left)}"></div>
    <div class="divider"></div>
    <span class="taga">{_esc(label_left)}</span><span class="tagb">{_esc(label_right)}</span>
    <input type="range" min="0" max="100" value="50" aria-label="Compare {_esc_attr(label_left)} and {_esc_attr(label_right)}">
  </div>{cap}'''


def expandable(label, body_html, open=False):
    """A v2 collapsed-by-default aside (2026-07-15, daily-report stack) for
    detail that supports a section but distracts from its message: method
    internals, provenance, derivations. Native <details>/<summary>, no JS.
    The summary renders as an editorial invitation row (mono kicker-style
    label between hairlines, accent disclosure glyph), not a boxed UI widget
    (that's v1 details_block's look). The body may hold breakout-width
    figures/diagrams; their own scroll boxes keep working when open.
    Prints collapsed, which this stack accepts. label is plain text in paper
    register ("How contact is detected"); no em dashes. Requires
    page(theme="v2")."""
    open_attr = " open" if open else ""
    return (f'<details class="expand"{open_attr}><summary>{_esc(label)}</summary>'
            f'<div class="expand-body">{body_html}</div></details>')


def annotated_figure(img_src, notes, alt="", caption_html=""):
    """A figure with numbered callout dots pinned onto the image. notes: list of
    {"x": <percent from left>, "y": <percent from top>, "text": <short HTML>}.
    Dots show their text as a hover tooltip (data-tip -> xg2.js) AND in a
    numbered list under the image, so the annotations survive print/touch."""
    dots = "".join(
        f'<div class="anno" style="left:{float(n["x"]):g}%;top:{float(n["y"]):g}%" '
        f'data-tip="{_esc_attr(n["text"])}">{i + 1}</div>'
        for i, n in enumerate(notes))
    items = "".join(
        f'<li><span class="anno-num">{i + 1}</span>{n["text"]}</li>'
        for i, n in enumerate(notes))
    cap = f'<figcaption>{caption_html}</figcaption>' if caption_html else ""
    return (f'<figure><div class="annofig">'
            f'<img src="{_esc_attr(img_src)}" alt="{_esc_attr(alt)}">{dots}</div>'
            f'<ol class="anno-list">{items}</ol>{cap}</figure>')


def hbar_chart(rows, *, title="", label_w=210, note="", aria=""):
    """A server-rendered horizontal bar chart in the v2 chart vocabulary —
    the v2 replacement for a table or <ul> of magnitudes. Bars carry hover
    tooltips (data-tip -> xg2.js); values are printed after each bar so the
    chart reads complete without hover/JS.
    - rows: list of dicts {"label": str, "value": float (bar length),
      "display": str shown after the bar (default str(value)),
      "tip": hover text (default display)}.
    - title: small axis-label line above the bars (state units + total there).
    - label_w: px reserved for row labels (raise for long labels).
    - note: optional .chartnote paragraph under the chart (raw HTML) — use it
      to state the finding, like a figcaption.
    Bars scale to max(value); zero/max-0 guarded."""
    n = len(rows)
    width = 720
    bar_x = label_w + 10
    bar_max = width - bar_x - 100
    top = 34 if title else 14
    height = top + n * 27 + 6
    max_v = max((float(r["value"]) for r in rows), default=0) or 1.0
    parts = []
    if title:
        parts.append(f'<text x="{bar_x}" y="20" class="axislabel">{_esc(title)}</text>')
    for i, r in enumerate(rows):
        y = top + i * 27
        w = max(round(float(r["value"]) / max_v * bar_max), 2)
        display = r.get("display", str(r["value"]))
        tip = r.get("tip", f'{r["label"]}: {display}')
        parts.append(f'<text x="{bar_x - 12}" y="{y + 13}" text-anchor="end" '
                     f'class="barlabel">{_esc(r["label"])}</text>')
        parts.append(f'<rect class="hbar" x="{bar_x}" y="{y}" width="{w}" height="18" '
                     f'rx="3" data-tip="{_esc_attr(tip)}"></rect>')
        parts.append(f'<text x="{bar_x + w + 6}" y="{y + 13}" class="barval">{_esc(display)}</text>')
    note_html = f'<p class="chartnote">{note}</p>' if note else ""
    aria_attr = _esc_attr(aria or title or "bar chart")
    return (f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{aria_attr}">{"".join(parts)}</svg>{note_html}</div>')


def chartnote(inner_html):
    """A short finding-stating paragraph under a chart or matplotlib PNG —
    same role as a figcaption, chart-flavored styling."""
    return f'<p class="chartnote">{inner_html}</p>'


def appendix(label, items, numbered=False):
    """A distill.pub-style end-matter block (2026-07-15, footnotes register,
    user reference: distill.pub's footnotes area) for de-emphasized material
    that belongs on the page but is not argued content: a daily log strip, a
    footnote list, provenance asides. NOT a section — no kicker number, no
    serif h2 — deliberately reads as end-matter, placed after the last
    section_v2() in body_sections. A full-width hairline rule separates it
    from the body; a small mono gutter LABEL ("Day log", "Appendix") sits
    top-aligned to the left of a compact list in smaller (~0.85em), muted
    (--ink-2) type with tight line-height. On phone widths the gutter label
    stacks above the list (theme2.css breakpoint).
    - label: short plain-text gutter label (escaped).
    - items: list of raw HTML strings — bold lead-in phrases are fine
      ("<b>Aesthetic rating evaluated at full scale.</b> ..."), but keep them
      visually quiet (no headings, no section numbers inside an item).
    - numbered: True for a mono "1." "2." … prefix (footnote-style); False
      (default) for a plain unnumbered list, the right choice when items
      aren't cross-referenced by number elsewhere on the page."""
    tag = "ol" if numbered else "ul"
    cls = "apx-list numbered" if numbered else "apx-list"
    lis = "".join(f"<li>{it}</li>" for it in items)
    return (f'<div class="appendix"><div class="apx-label">{_esc(label)}</div>'
            f'<{tag} class="{cls}">{lis}</{tag}></div>')
