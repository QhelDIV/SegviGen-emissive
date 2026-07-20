"""
Build a LIVE curated project index for the lightgen visuals on aspis.

Writes two files into <web_root>/<project>/:
  - manifest.json : the data (entries with title/brief/status + last-updated mtime)
  - index.html    : a self-refreshing page that fetches manifest.json every 60s,
                    badges NEW / UPDATED pages, and fires a desktop notification +
                    sound when a page is added/updated while you have it open.

Re-run after publishing a page (or wire into the publish step). Status/brief come
from web_index.json; the per-page "updated" time is the mtime of each page's index.html.

  python build_web_index.py [web_index.json]
"""
import os, sys, json, html, time, datetime

WEB_ROOT = "/project/3dlg-hcvc/omages/www/yanxg"
URLBASE  = "https://aspis.cmpt.sfu.ca/projects/omages/yanxg"


def main():
    man = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "web_index.json"))
    proj = man["project"]
    pdir = os.path.join(WEB_ROOT, proj)
    os.makedirs(pdir, exist_ok=True)

    curated = {e["slug"]: e for e in man["entries"]}
    # auto-discover any published page under the project not yet in the manifest, so new
    # publishes always appear (and trigger the alarm) even before a brief is written.
    discovered = []
    for name in sorted(os.listdir(pdir)):
        sub = os.path.join(pdir, name)
        if name not in curated and os.path.isfile(os.path.join(sub, "index.html")):
            discovered.append({"slug": name, "title": name,
                               "brief": "(auto-discovered — no description yet)", "status": "live"})
    all_e = man["entries"] + discovered

    entries = []
    for e in all_e:
        slug = e["slug"]
        ip = os.path.join(pdir, slug, "index.html")
        exists = os.path.isfile(ip)
        mtime = int(os.path.getmtime(ip)) if exists else 0
        status = e.get("status", "live")
        if not exists and status in ("live", "rendering"):
            status = "missing"
        entries.append({"slug": slug, "title": e["title"], "brief": e.get("brief", ""),
                        "status": status, "exists": exists, "mtime": mtime,
                        "url": f"{URLBASE}/{proj}/{slug}/index.html"})
    entries.sort(key=lambda x: x["mtime"], reverse=True)
    if discovered:
        print(f"auto-discovered {len(discovered)} page(s): {[d['slug'] for d in discovered]}")

    manifest = {"project": proj, "title": man["title"],
                "generated": int(time.time()), "entries": entries}
    with open(os.path.join(pdir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    os.chmod(os.path.join(pdir, "manifest.json"), 0o644)

    doc = """<!DOCTYPE html><html><head><meta charset=utf-8><title>__TITLE__</title><style>
body{background:#0e1117;color:#d8dde6;font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:32px auto;padding:0 22px}
h1{font-size:21px;margin:0 0 4px} .sub{color:#8b949e;font-size:13px;margin-bottom:14px}
.bar{display:flex;gap:10px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
button{background:#161b22;color:#d8dde6;border:1px solid #30363d;border-radius:7px;padding:6px 12px;font-size:13px;cursor:pointer}
button:hover{border-color:#58a6ff} #alert{display:none;background:#1f6feb22;border:1px solid #1f6feb;color:#cae3ff;padding:8px 12px;border-radius:8px;font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:#8b949e;font-weight:600;font-size:12px;border-bottom:1px solid #30363d;padding:6px 10px}
td{padding:11px 10px;border-bottom:1px solid #21262d;vertical-align:top}
.t a{color:#58a6ff;text-decoration:none;font-weight:600} .t a:hover{text-decoration:underline}
.slug{font-family:ui-monospace,monospace;font-size:11px;color:#6e7681;margin-top:3px}
.brief{color:#b3bcc6;font-size:13px;line-height:1.45;max-width:560px}
.badge{color:#fff;font-size:11px;padding:2px 9px;border-radius:10px;white-space:nowrap}
.flag{font-size:10px;font-weight:700;padding:2px 7px;border-radius:9px;margin-left:8px}
.new{background:#2ea043;color:#fff} .upd{background:#9e6a03;color:#fff}
tr.changed td{background:#13251a}
</style></head><body>
<h1>__TITLE__</h1>
<div class=sub>Live index · auto-checks every 5s · <a href="__URLBASE__/" style="color:#58a6ff">↑ all projects</a> ·
<span id=gen></span></div>
<div class=bar>
  <button id=notify>🔔 Enable desktop alerts</button>
  <button id=seen>Mark all seen</button>
  <span id=alert></span>
</div>
<table><tr><th>page</th><th>brief</th><th>status</th></tr><tbody id=tb></tbody></table>
<script>
const URL="__URLBASE__/__PROJ__/manifest.json";
const COLOR={live:"#2ea043",rendering:"#9e6a03",superseded:"#6e7681",deprecated:"#8b3a3a",missing:"#8b3a3a"};
const esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
let prev=null;  // last manifest seen by THIS tab (for live-change detection)
function seenMap(){try{return JSON.parse(localStorage.getItem("lg_seen")||"{}")}catch(e){return {}}}
function setSeen(m){localStorage.setItem("lg_seen",JSON.stringify(m))}
function beep(){try{const a=new(window.AudioContext||window.webkitAudioContext)();const o=a.createOscillator(),g=a.createGain();o.connect(g);g.connect(a.destination);o.frequency.value=880;g.gain.value=0.07;o.start();o.stop(a.currentTime+0.18)}catch(e){}}
function render(man){
  const seen=seenMap(); let nNew=0,nUpd=0;
  document.getElementById("gen").textContent="updated "+new Date(man.generated*1000).toLocaleString();
  document.getElementById("tb").innerHTML=man.entries.map(e=>{
    let flag="";
    if(!(e.slug in seen)){if(e.exists){flag='<span class="flag new">NEW</span>';nNew++}}
    else if(seen[e.slug]<e.mtime){flag='<span class="flag upd">UPDATED</span>';nUpd++}
    const link=e.exists?`<a href="${e.url}">${esc(e.title)}</a>`:esc(e.title);
    const col=COLOR[e.status]||"#6e7681";
    return `<tr class="${flag?'changed':''}"><td class=t>${link}${flag}<div class=slug>${esc(e.slug)}</div></td>
      <td class=brief>${esc(e.brief)}</td><td><span class=badge style="background:${col}">${e.status}</span></td></tr>`;
  }).join("");
  const a=document.getElementById("alert");
  if(nNew+nUpd>0){a.style.display="inline-block";a.textContent=`${nNew} new · ${nUpd} updated since you last marked seen`;}
  else{a.style.display="none";}
}
function liveDiff(man){
  // fire alarm only for changes detected WHILE this tab is open (vs previous poll)
  if(!prev) return [];
  const pm={}; prev.entries.forEach(e=>pm[e.slug]=e.mtime);
  return man.entries.filter(e=>e.exists && (!(e.slug in pm) || pm[e.slug]<e.mtime));
}
async function poll(){
  try{
    const man=await (await fetch(URL+"?t="+Date.now(),{cache:"no-store"})).json();
    const changed=liveDiff(man);
    render(man);
    if(changed.length){
      beep();
      const names=changed.map(e=>e.title).join(", ");
      if(window.Notification && Notification.permission==="granted")
        new Notification("lightgen visuals updated",{body:changed.length+" page(s): "+names});
      document.title="● "+changed.length+" new — "+BASE_TITLE;
    }
    prev=man;
  }catch(e){}
}
const BASE_TITLE="__TITLE__";
document.getElementById("notify").onclick=()=>{if(window.Notification)Notification.requestPermission().then(p=>{if(p==="granted")new Notification("Alerts enabled","")})};
document.getElementById("seen").onclick=()=>{const m={};(prev?prev.entries:[]).forEach(e=>m[e.slug]=e.mtime);setSeen(m);document.title=BASE_TITLE;render(prev)};
poll(); setInterval(poll,5000);
</script></body></html>"""
    doc = (doc.replace("__TITLE__", html.escape(man["title"]))
              .replace("__URLBASE__", URLBASE).replace("__PROJ__", proj))
    out = os.path.join(pdir, "index.html")
    open(out, "w").write(doc)
    os.chmod(out, 0o644)
    print(f"wrote {out} + manifest.json  ({len(entries)} entries)")
    print(f"{URLBASE}/{proj}/index.html")


if __name__ == "__main__":
    main()
