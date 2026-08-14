"""
Build a self-contained HTML dashboard from a Claude Code session transcript (JSONL).
Shows token usage, cost estimate, hourly activity timeline, tool histogram, and a
per-turn table — for reviewing a long (e.g. overnight autonomous) session.

Usage: python session_dashboard.py <session.jsonl> [more.jsonl ...] -o out.html
"""
import sys, json, argparse, html, datetime as dt
from collections import Counter, defaultdict

# Opus pricing ($ / 1M tokens) — ESTIMATE; cross-check with /cost.
PRICE = {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50}


def parse(paths):
    turns = []
    for p in paths:
        for line in open(p, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("type") != "assistant":
                continue
            msg = o.get("message", {})
            u = msg.get("usage", {}) or {}
            tools, text = [], ""
            for b in (msg.get("content") or []):
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    tools.append(b.get("name", "?"))
                elif b.get("type") == "text":
                    text += b.get("text", "")
            stu = u.get("server_tool_use", {}) or {}
            turns.append({
                "ts": o.get("timestamp"),
                "model": msg.get("model", "?"),
                "in": u.get("input_tokens", 0),
                "out": u.get("output_tokens", 0),
                "cw": u.get("cache_creation_input_tokens", 0),
                "cr": u.get("cache_read_input_tokens", 0),
                "web": stu.get("web_search_requests", 0) + stu.get("web_fetch_requests", 0),
                "tools": tools,
                "text": text.strip(),
            })
    turns = [t for t in turns if t["ts"]]
    turns.sort(key=lambda t: t["ts"])
    return turns


def cost(t):
    return (t["in"] * PRICE["input"] + t["out"] * PRICE["output"]
            + t["cw"] * PRICE["cache_write"] + t["cr"] * PRICE["cache_read"]) / 1e6


def fmt(n):
    return f"{n:,}"


def build(turns, out):
    if not turns:
        open(out, "w").write("<h1>No assistant turns found.</h1>"); return
    t0 = dt.datetime.fromisoformat(turns[0]["ts"].replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(turns[-1]["ts"].replace("Z", "+00:00"))
    span_h = (t1 - t0).total_seconds() / 3600
    tot = {k: sum(t[k] for t in turns) for k in ["in", "out", "cw", "cr", "web"]}
    tot_cost = sum(cost(t) for t in turns)
    tool_hist = Counter(name for t in turns for name in t["tools"])
    n_tools = sum(tool_hist.values())

    # ---- time-series over WALL-CLOCK time (x = real timestamp, y = tokens) ----
    def epoch(ts): return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    t0s, t1s = epoch(turns[0]["ts"]), epoch(turns[-1]["ts"])
    tspan = max(1.0, t1s - t0s)

    def human(v):
        v = float(v)
        return f"{v/1e6:.1f}M" if v >= 1e6 else (f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")

    cum = 0; cum_pts = []
    for t in turns:
        cum += t["out"]; cum_pts.append((epoch(t["ts"]), cum))
    ymax_cum = cum or 1

    rate = defaultdict(int)
    for t in turns:
        rate[int(epoch(t["ts"]) // 3600 * 3600)] += t["out"]
    rmax = max(rate.values(), default=1)

    def axes_svg(width, height, draw_fn, ymax, ylabel):
        ml, mr, mt, mb = 80, 18, 14, 50
        pw, ph = width - ml - mr, height - mt - mb
        def X(ts): return ml + (ts - t0s) / tspan * pw
        def Y(v):  return mt + ph - (v / ymax) * ph
        s = [f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#0b0f14" stroke="#2b313a"/>']
        for i in range(6):                                  # y gridlines + labels
            v = ymax * i / 5; y = Y(v)
            s.append(f'<line x1="{ml}" y1="{y:.0f}" x2="{ml+pw}" y2="{y:.0f}" stroke="#1b2026"/>')
            s.append(f'<text x="{ml-8}" y="{y+3:.0f}" font-size="10" fill="#8b949e" text-anchor="end">{human(v)}</text>')
        for i in range(8):                                  # x ticks (time) + labels
            ts = t0s + tspan * i / 7; x = X(ts)
            s.append(f'<line x1="{x:.0f}" y1="{mt}" x2="{x:.0f}" y2="{mt+ph}" stroke="#161b22"/>')
            lbl = dt.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
            s.append(f'<text x="{x:.0f}" y="{mt+ph+14}" font-size="9" fill="#8b949e" text-anchor="end" transform="rotate(-35 {x:.0f} {mt+ph+14})">{lbl}</text>')
        s.append(f'<text x="16" y="{mt+ph/2:.0f}" font-size="11" fill="#c9d1d9" text-anchor="middle" transform="rotate(-90 16 {mt+ph/2:.0f})">{ylabel}</text>')
        s.append(f'<text x="{ml+pw/2:.0f}" y="{height-4}" font-size="11" fill="#c9d1d9" text-anchor="middle">wall-clock time</text>')
        s.append(draw_fn(X, Y))
        return f'<svg width="{width}" height="{height}">' + "".join(s) + '</svg>'

    def cum_draw(X, Y):
        pts = " ".join(f"{X(ts):.1f},{Y(v):.1f}" for ts, v in cum_pts)
        area = f"{X(cum_pts[0][0]):.1f},{Y(0):.1f} {pts} {X(cum_pts[-1][0]):.1f},{Y(0):.1f}"
        return (f'<polygon points="{area}" fill="#4e9af122"/>'
                f'<polyline points="{pts}" fill="none" stroke="#4e9af1" stroke-width="2"/>')

    def rate_draw(X, Y):
        bw = max(1.5, 3600 / tspan * 802)
        return "".join(
            f'<rect x="{X(hr):.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{Y(0)-Y(v):.1f}" '
            f'fill="#7c5cff"><title>{dt.datetime.fromtimestamp(hr).strftime("%m-%d %H:00")}: {human(v)}</title></rect>'
            for hr, v in sorted(rate.items()))

    cum_svg = axes_svg(900, 300, cum_draw, ymax_cum, "cumulative output tokens")
    rate_svg = axes_svg(900, 240, rate_draw, rmax, "output tokens / hour")

    def card(label, val, sub=""):
        return (f'<div class="card"><div class="cv">{val}</div>'
                f'<div class="cl">{label}</div>{f"<div class=cs>{sub}</div>" if sub else ""}</div>')

    cards = "".join([
        card("turns", fmt(len(turns))),
        card("output tokens", fmt(tot["out"])),
        card("cache-read tokens", fmt(tot["cr"]), "context replays"),
        card("cache-write tokens", fmt(tot["cw"])),
        card("input tokens", fmt(tot["in"])),
        card("est. cost", f"${tot_cost:,.2f}", "Opus rates — verify /cost"),
        card("tool calls", fmt(n_tools)),
        card("web fetch/search", fmt(tot["web"])),
        card("span", f"{span_h:.1f} h", f'{t0.strftime("%m-%d %H:%M")} → {t1.strftime("%H:%M")}'),
    ])

    # tool histogram (CSS bars)
    tmax = max(tool_hist.values(), default=1)
    th = "".join(
        f'<div class="trow"><div class="tn">{html.escape(n)}</div>'
        f'<div class="tb"><div class="tbf" style="width:{c/tmax*100:.0f}%"></div></div>'
        f'<div class="tc">{c}</div></div>'
        for n, c in tool_hist.most_common())

    # per-turn table (recent text snippet)
    rows = ""
    for i, t in enumerate(turns):
        tm = dt.datetime.fromisoformat(t["ts"].replace("Z", "+00:00")).strftime("%m-%d %H:%M:%S")
        snip = html.escape(t["text"][:140].replace("\n", " "))
        tl = html.escape(", ".join(t["tools"])[:60])
        rows += (f'<tr><td>{i+1}</td><td class=mono>{tm}</td><td>{fmt(t["out"])}</td>'
                 f'<td>{fmt(t["cr"])}</td><td>{tl}</td><td class=snip>{snip}</td></tr>')

    doc = f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Session dashboard</title><style>
body{{background:#0e1117;color:#d8dde6;font-family:-apple-system,Segoe UI,Helvetica,sans-serif;margin:0;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8b949e;font-size:13px;margin-bottom:20px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:28px}}
.card{{background:#161b22;border:1px solid #2b313a;border-radius:10px;padding:14px 18px;min-width:130px}}
.cv{{font-size:24px;font-weight:700;color:#fff}} .cl{{font-size:12px;color:#9aa4b2;margin-top:2px}} .cs{{font-size:10px;color:#6b7280;margin-top:2px}}
h2{{font-size:15px;color:#c9d1d9;margin:24px 0 10px;border-bottom:1px solid #2b313a;padding-bottom:6px}}
.legend{{font-size:12px;color:#9aa4b2;margin-bottom:8px}} .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle}}
.chart{{overflow-x:auto;background:#161b22;border:1px solid #2b313a;border-radius:10px;padding:12px}}
.trow{{display:flex;align-items:center;gap:10px;margin:4px 0;font-size:13px}} .tn{{width:170px;color:#c9d1d9}} .tb{{flex:1;background:#21262d;border-radius:4px;height:16px}} .tbf{{height:16px;background:#4e9af1;border-radius:4px}} .tc{{width:48px;text-align:right;color:#9aa4b2}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}} th,td{{padding:5px 8px;border-bottom:1px solid #21262d;text-align:left;vertical-align:top}} th{{color:#8b949e;position:sticky;top:0;background:#0e1117}}
.mono{{font-family:ui-monospace,monospace;color:#9aa4b2;white-space:nowrap}} .snip{{color:#8b949e}} .tbl{{max-height:420px;overflow:auto;border:1px solid #2b313a;border-radius:10px}}
</style></head><body>
<h1>Claude Code session dashboard</h1>
<div class=sub>{len(turns)} assistant turns · {t0.strftime("%Y-%m-%d %H:%M")} → {t1.strftime("%H:%M")} ({span_h:.1f} h)</div>
<div class=cards>{cards}</div>
<h2>Cumulative output tokens over wall-clock time</h2>
<div class=chart>{cum_svg}</div>
<h2>Output tokens per hour (wall-clock)</h2>
<div class=chart>{rate_svg}</div>
<h2>Tool usage</h2>
{th}
<h2>Per-turn log</h2>
<div class=tbl><table><tr><th>#</th><th>time</th><th>out tok</th><th>cache-rd</th><th>tools</th><th>text snippet</th></tr>{rows}</table></div>
</body></html>"""
    open(out, "w", encoding="utf-8").write(doc)
    print(f"wrote {out}  ({len(turns)} turns, {fmt(tot['out'])} out tok, ~${tot_cost:,.2f}, {span_h:.1f}h)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("-o", "--out", default="session_dashboard.html")
    a = ap.parse_args()
    build(parse(a.jsonl), a.out)
