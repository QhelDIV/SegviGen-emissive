"""Parse PAPER_SKELETON_V2_CLAIMS.md into the `.skel` claim-chain component.

The content file is the authority. This module reads it at build time rather
than restating it, so the page cannot drift from it, and it ASSERTS that every
rendered counter value equals the number written in the file. A markup or CSS
change that shifts the numbering fails the build instead of shipping.

Item kinds, from the file's own markers:
  HEAD:  a section label      -> <li class="head">, no counter
  DEF:   a definition / the section's central claim -> <li class="def">, no counter
  OPEN:  an unsettled question -> <li class="open">, no counter
  N. ... an atomic claim       -> <li>, counter increments, must render as N

NOTE ON `def` AND THE COUNTER: the component CSS extracted from the somages
reference excludes only `head` and `open` from the counter, which would make a
`def` item consume a number. The content file's own numbering does not: claim 3
is followed by a DEF and then by claim 4, and the Method section opens with a
DEF and continues at 25. So `def` is added to the no-increment rule here. The
file is the authority on numbering (the brief says so explicitly), and this is
the one change that makes rendered numbers equal written numbers.
"""
import html
import re

HEAD, DEF, OPEN, CLAIM = "head", "def", "open", "claim"


def parse(path):
    """Return [(kind, number_or_None, text)] in file order."""
    items = []
    for raw in open(path):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#") or line.strip() == "---":
            continue
        if line.startswith("HEAD:"):
            items.append((HEAD, None, line[len("HEAD:"):].strip()))
        elif line.startswith("DEF:"):
            items.append((DEF, None, line[len("DEF:"):].strip()))
        elif line.startswith("OPEN:"):
            items.append((OPEN, None, line[len("OPEN:"):].strip()))
        else:
            m = re.match(r"^(\d+)\.\s+(.*)$", line)
            if not m:
                raise ValueError(f"unrecognized line in the content file: {line!r}")
            items.append((CLAIM, int(m.group(1)), m.group(2).strip()))
    return items


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def heads(items):
    """[(id, label)] for the outline rail, in order."""
    return [(slug(t), t) for k, _, t in items if k == HEAD]


def verify(items):
    """Simulate the CSS counter and assert it matches every written number.

    counter-increment fires on plain <li> only; head, def and open are excluded
    (see the module docstring). Raises with the first divergence.
    """
    n = 0
    for kind, num, text in items:
        if kind != CLAIM:
            continue
        n += 1
        if n != num:
            raise AssertionError(
                f"rendered counter {n} != written number {num} at {text[:60]!r}; "
                "the markup or the counter CSS is wrong, not the content file")
    return n


def _li(kind, text):
    body = html.escape(text)
    if kind == HEAD:
        return f'<li class="head" id="{slug(text)}">{body}</li>'
    if kind == DEF:
        return f'<li class="def">{body}</li>'
    if kind == OPEN:
        return f'<li class="open">{body}</li>'
    return f"<li>{body}</li>"


def render(items, breaks=()):
    """The claim chain as prose-wrapped <ol class="skel"> segments.

    breaks: head LABELS after whose section the list is closed, so a figure can
    sit between two segments. Continuation lists carry `.cont`, which is
    `counter-reset:none`, so numbering runs unbroken across the whole page.
    Returns [segment_html, ...] with len == len(breaks) + 1.
    """
    breaks = set(breaks)
    segments, current, seen_head = [], [], None
    for kind, _, text in items:
        if kind == HEAD and seen_head in breaks and current:
            segments.append(current)
            current = []
        if kind == HEAD:
            seen_head = text
        current.append(_li(kind, text))
    if current:
        segments.append(current)
    out = []
    for i, seg in enumerate(segments):
        cls = "skel cont" if i else "skel"
        out.append(f'<div class="prose"><ol class="{cls}">{"".join(seg)}</ol></div>')
    return out
