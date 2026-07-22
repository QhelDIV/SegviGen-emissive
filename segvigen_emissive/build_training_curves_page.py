"""
Assemble vis_data/training_curves_html/index.html — loss + quick-val IoU curves for
both Phase 4 fine-tune arms (W5 pos_weight=5 "eager" vs W1 pos_weight=1 "timid"), plus
the overfit-1/overfit-10 sanity gates that preceded them, and the full val_96 4-checkpoint
result table.

2026-07-06: this page was originally built by a different session with no local build
script checked in here — only the published HTML existed. Ported onto tools/xgpage.py
(the shared design-system component module) by extracting the live page's exact HTML/
data/JS verbatim (curl'd from the published URL) and re-composing it into the new
Overview -> Curves -> Gates -> Final-results report structure (outline + Medium-style
preview/expand), same as finetune_binary_v1 and dataset_gallery_v1. Every number, caption,
and the entire interactive-chart JS (DATA + drawChart()) are unchanged from the original —
this is re-plumbing, not re-analysis.

The charts here are CLIENT-SIDE interactive (hover crosshair + tooltip), drawn at runtime
by drawChart() from the embedded DATA object — unlike dataset_gallery_v1's server-rendered
static SVG strings. That JS is page-specific (not a reusable xgpage.py component; only
this page needs multi-series hover line charts) and is kept verbatim as one script block.

  python build_training_curves_page.py
"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "vis_data", "training_curves_html")
os.makedirs(OUT, exist_ok=True)

import xgpage as lp  # the installed package (uv pip install -e ~/studio/xgpage); migrated 2026-07-22

# Cluster path root (verified 2026-07-07 via `ssh solar` directory listing) for the
# filepath() hover/click-copy component.
SEGVIGEN_ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive"

# ---------------------------------------------------------------------------------
# Page-specific palette + a few page-specific size tweaks, layered on top of theme.css.
# This page keeps its own 5-way categorical scheme (W5/W1/two gate colors/reference-line
# grey) rather than remapping onto the shared --accent/--accent2 names, because its
# .tag.w5/.tag.w1 rules here use different literal hex values (blue/orange) than the
# cyan/amber used by finetune_binary_v1 and dataset_gallery_v1's --accent/--accent2 —
# forcing them onto the same variable names would either recolor those other pages or
# require per-page overrides of the shared .tag rules anyway. Kept as an intentional,
# documented diff (see the extraction report) rather than a silent inconsistency.
PAGE_STYLE = """
  :root {
    --w5: #3987e5;      /* categorical slot 1 — pos_weight=5.0 "eager" */
    --w1: #d95926;      /* categorical slot 8 — pos_weight=1.0 "timid" control */
    --aux1: #199e70;    /* categorical slot 2 — overfit-10 gate -> extension */
    --aux2: #c98500;    /* categorical slot 3 — amended gate (train_1k_gate10) */
    --ref: #6b7480;     /* reference/threshold lines — not a data series */
  }
  .page { max-width: 1320px; }
  h1 { font-size: 1.6rem; }
  h2 .num { color: var(--w5); }
  a { color: #7db8f5; }
  code { color: var(--w5); }
  .callout { border-left-color: var(--w5); }
  .tag.w5 { color: var(--w5); border-color: rgba(57,135,229,.4); }
  .tag.w1 { color: var(--w1); border-color: rgba(217,89,38,.4); }
  .legend span.ln { border-top-color: var(--ref); }
"""

# ---------------------------------------------------------------------------------
# Raw per-epoch data, verbatim from the live page (parsed from Slurm logs /
# train_curve.json on the cluster by the session that originally built this page).
DATA_JS = """{"w5_flow_loss": [[1, 0.2737775388176247], [2, 0.24999766271185703], [3, 0.25747427487047597], [4, 0.25343769742321065], [5, 0.24302114937197375], [6, 0.26734344351029926], [7, 0.25725640184353815], [8, 0.2616463305224612], [9, 0.23520953881468057], [10, 0.26183152833156526], [11, 0.2747068353246513], [12, 0.26080645963413607], [13, 0.2625202552165806], [14, 0.24259003322573175], [15, 0.26582871088326865], [16, 0.2511115988191111], [17, 0.24533788405171367], [18, 0.25716961655895976]], "w1_flow_loss": [[1, 0.24529832420673284], [2, 0.2463717749047915], [3, 0.23497020403372715], [4, 0.24552594804418965], [5, 0.2562796082882541], [6, 0.2271008355124627], [7, 0.2356428952175686], [8, 0.23333476492944352], [9, 0.23344120603943788], [10, 0.21585283747489398], [11, 0.240568180698097], [12, 0.22770884275421474], [13, 0.24016395386412925], [14, 0.22540229309589924], [15, 0.2431242672514207], [16, 0.22063425025871175], [17, 0.22438452289670288], [18, 0.2345443505882754]], "w5_val_iou": [[2, 0.09646921036187428], [4, 0.12775939108421713], [6, 0.09730179543908414], [8, 0.09857481528076358], [10, 0.09975036488785088], [12, 0.0886040050791409], [14, 0.2151295345614728], [16, 0.13315835711166255], [18, 0.04022903669388112]], "w1_val_iou": [[2, 0.10500377390539463], [4, 0.12320945783107412], [6, 0.1260027490754494], [8, 0.10646684733118741], [10, 0.1140000958302224], [12, 0.08432655941503478], [14, 0.13064936964044727], [16, 0.22061779576883694], [18, 0.11664016625289883]], "overfit1_w5_iou": [[5, 0.41], [10, 0.4175], [15, 0.4226], [20, 0.426], [25, 0.2851], [30, 0.885]], "overfit1_w1_iou": [[5, 0.0752], [10, 0.4114], [15, 0.4173], [20, 0.4195], [25, 0.4258], [30, 0.9177]], "overfit10_gate_iou": [[5, 0.1317], [10, 0.1455], [15, 0.1601], [20, 0.208], [25, 0.1746], [30, 0.1703], [35, 0.2908], [40, 0.253]], "overfit10_ext_iou": [[50, 0.2102], [60, 0.2497], [70, 0.3565], [80, 0.2075], [90, 0.3137], [100, 0.2303], [110, 0.2662], [120, 0.3519], [130, 0.3266], [140, 0.3694], [150, 0.291], [160, 0.2841], [170, 0.295], [180, 0.3622], [190, 0.2983], [200, 0.3957], [210, 0.4184], [220, 0.3704], [230, 0.297], [240, 0.3313], [250, 0.3561], [260, 0.3208], [270, 0.4762], [280, 0.3946], [290, 0.3549], [300, 0.3776]], "amended_gate_iou": [[10, 0.2511], [20, 0.2509], [30, 0.3633], [40, 0.3719], [50, 0.3268], [60, 0.322], [70, 0.3718], [80, 0.2858], [90, 0.384], [100, 0.4397], [110, 0.4557], [120, 0.3921], [130, 0.3849], [140, 0.4359], [150, 0.5542]]}"""

# ---------------------------------------------------------------------------------
# The interactive chart engine + all 4 chart invocations, verbatim from the live page
# (generic SVG line-chart drawer with running-mean, reference lines, best-epoch labels,
# and a hover crosshair+tooltip — see dataviz skill re: the hover-layer convention).
CHART_SCRIPT = """
<script>
const DATA = """ + DATA_JS + """;

// ---------- generic SVG line chart ----------
function fmt2(v){ return v.toFixed(2); }
function fmt3(v){ return v.toFixed(3); }

function drawChart(svgId, cfg){
  const svg = document.getElementById(svgId);
  const NS = 'http://www.w3.org/2000/svg';
  const W = cfg.width, H = cfg.height;
  const m = cfg.margin;
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const [x0,x1] = cfg.xDomain, [y0,y1] = cfg.yDomain;
  const xs = v => m.left + (v - x0) / (x1 - x0) * iw;
  const ys = v => m.top + ih - (v - y0) / (y1 - y0) * ih;

  function el(tag, attrs, parent){
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  }

  // gridlines + y ticks
  cfg.yTicks.forEach(t => {
    el('line', {x1:m.left, x2:m.left+iw, y1:ys(t), y2:ys(t), stroke:'#242a34', 'stroke-width':1});
    el('text', {x:m.left-8, y:ys(t)+4, 'text-anchor':'end', 'font-size':11, fill:'#8b96a5'}).textContent = cfg.yFormat ? cfg.yFormat(t) : t;
  });
  // x axis
  el('line', {x1:m.left, x2:m.left+iw, y1:m.top+ih, y2:m.top+ih, stroke:'#333b47', 'stroke-width':1});
  cfg.xTicks.forEach(t => {
    el('text', {x:xs(t), y:m.top+ih+20, 'text-anchor':'middle', 'font-size':11, fill:'#8b96a5'}).textContent = 'ep ' + t;
    el('line', {x1:xs(t), x2:xs(t), y1:m.top+ih, y2:m.top+ih+4, stroke:'#333b47', 'stroke-width':1});
  });
  if (cfg.yAxisLabel) {
    el('text', {x:m.left, y:14, 'font-size':11, fill:'#6b7480'}).textContent = cfg.yAxisLabel;
  }

  // reference lines — draw the dashed lines at their true y, but lay out the
  // text labels with a minimum vertical gap so close-valued lines don't collide
  const refs = (cfg.refLines || []).map(r => ({...r, ly: ys(r.y)}));
  refs.forEach(r => {
    el('line', {x1:m.left, x2:m.left+iw, y1:ys(r.y), y2:ys(r.y), stroke:'var(--ref)', 'stroke-width':1.5, 'stroke-dasharray':'5,4'});
  });
  // when reference lines sit close together, stack ALL their labels above the
  // topmost line (never let a label straddle its own or a neighbor's dashes)
  refs.sort((a,b) => a.ly - b.ly);
  const REF_GAP = 14;
  const topLy = refs.length ? refs[0].ly : 0;
  refs.forEach((r,i) => { r.labelY = topLy - 5 - i*REF_GAP; });
  const refAnchor = cfg.refLabelAnchor || 'end';
  const refX = refAnchor === 'start' ? m.left+4 : m.left+iw-4;
  refs.forEach(r => {
    el('text', {x:refX, y:r.labelY, 'text-anchor':refAnchor, 'font-size':11, fill:'#9aa3af'}).textContent = r.label;
  });

  function path(pts){
    return pts.map((p,i) => (i===0?'M':'L') + xs(p[0]).toFixed(1) + ' ' + ys(p[1]).toFixed(1)).join(' ');
  }
  function runningMean(pts, win){
    return pts.map((p,i) => {
      const lo = Math.max(0, i-Math.floor(win/2)), hi = Math.min(pts.length, i+Math.ceil(win/2));
      const slice = pts.slice(lo,hi);
      const m = slice.reduce((s,q)=>s+q[1],0)/slice.length;
      return [p[0], m];
    });
  }

  const seriesLayers = [];
  cfg.series.forEach(s => {
    if (s.showRaw !== false) {
      el('path', {d:path(s.points), fill:'none', stroke:s.color, 'stroke-width': s.raw ? 1.3 : 2,
                  'stroke-opacity': s.raw ? 0.45 : 1, 'stroke-linejoin':'round', 'stroke-linecap':'round'});
    }
    if (s.mean) {
      const mp = runningMean(s.points, s.meanWindow || 3);
      el('path', {d:path(mp), fill:'none', stroke:s.color, 'stroke-width':2.5, 'stroke-linejoin':'round', 'stroke-linecap':'round'});
    }
    if (s.dots !== false) {
      s.points.forEach(p => {
        el('circle', {cx:xs(p[0]), cy:ys(p[1]), r: (s.markBest && p[0]===s.bestX) ? 6 : 3.4,
                       fill: (s.markBest && p[0]===s.bestX) ? s.color : s.color,
                       stroke: cfg.surface || '#171b22', 'stroke-width':2});
      });
    }
    seriesLayers.push(s);
  });

  // best-epoch text labels: collect candidates first, then de-collide vertically
  // (two series can peak at the same/nearby epoch with close values — see overfit-1)
  const bestLabels = [];
  cfg.series.forEach(s => {
    if (s.markBest && s.bestX != null && s.noBestLabel !== true) {
      const bp = s.points.find(p=>p[0]===s.bestX);
      if (bp) bestLabels.push({x:xs(bp[0]), y:ys(bp[1])-12, color:s.color, text:'best ' + fmt3(bp[1]) + ' @ep' + bp[0]});
    }
  });
  bestLabels.sort((a,b) => a.x - b.x || a.y - b.y);
  const LABEL_XGAP = 70, LABEL_YGAP = 13;
  // clear any reference-line dashes first (a "best" label must never sit on a threshold line)
  const refLineYs = refs.map(r => r.ly);
  const REF_CLEAR = 9;
  if (refLineYs.length) {
    const minRefLy = Math.min(...refLineYs), maxRefLy = Math.max(...refLineYs);
    bestLabels.forEach(b => {
      if (b.y > minRefLy - REF_CLEAR && b.y < maxRefLy + REF_CLEAR) b.y = minRefLy - REF_CLEAR;
    });
  }
  for (let i=1;i<bestLabels.length;i++){
    for (let j=0;j<i;j++){
      if (Math.abs(bestLabels[i].x - bestLabels[j].x) < LABEL_XGAP && Math.abs(bestLabels[i].y - bestLabels[j].y) < LABEL_YGAP) {
        bestLabels[i].y = bestLabels[j].y - LABEL_YGAP;
      }
    }
  }
  bestLabels.forEach(b => {
    el('text', {x:b.x, y:b.y, 'text-anchor': b.x > m.left+iw-90 ? 'end':'middle', 'font-size':11, fill:b.color, 'font-weight':600})
      .textContent = b.text;
  });

  // hover crosshair + tooltip
  const hover = el('g', {opacity:0});
  const vline = el('line', {y1:m.top, y2:m.top+ih, stroke:'#3a4250', 'stroke-width':1, 'stroke-dasharray':'3,3'}, hover);
  const hoverDots = seriesLayers.map(s => el('circle', {r:5, fill:s.color, stroke:'#fff', 'stroke-width':1.4}, hover));

  const wrap = svg.closest('.diagram-wrap');
  let tt = wrap.querySelector('.chart-tt');
  if (!tt) { tt = document.createElement('div'); tt.className='chart-tt'; wrap.style.position='relative'; wrap.appendChild(tt); }

  const overlay = el('rect', {x:m.left, y:m.top, width:iw, height:ih, fill:'transparent'});
  overlay.addEventListener('mousemove', (ev) => {
    const rect = svg.getBoundingClientRect();
    const scaleX = W / rect.width;
    const mx = (ev.clientX - rect.left) * scaleX;
    const xv = x0 + (mx - m.left) / iw * (x1 - x0);
    // find nearest x among all series' actual points
    let allX = [];
    seriesLayers.forEach(s => s.points.forEach(p => allX.push(p[0])));
    allX = [...new Set(allX)];
    const nearest = allX.reduce((a,b) => Math.abs(b-xv) < Math.abs(a-xv) ? b : a);
    hover.setAttribute('opacity', 1);
    vline.setAttribute('x1', xs(nearest)); vline.setAttribute('x2', xs(nearest));
    let rows = '';
    seriesLayers.forEach((s, i) => {
      const p = s.points.find(q => q[0] === nearest);
      if (p) {
        hoverDots[i].setAttribute('cx', xs(p[0])); hoverDots[i].setAttribute('cy', ys(p[1]));
        hoverDots[i].setAttribute('opacity', 1);
        rows += '<div class="row"><span class="d" style="background:'+s.color+'"></span>'+s.label+': <b style="margin:0">'+fmt3(p[1])+'</b></div>';
      } else {
        hoverDots[i].setAttribute('opacity', 0);
      }
    });
    tt.innerHTML = '<b>epoch ' + nearest + '</b>' + rows;
    tt.style.opacity = 1;
    const wrapRect = wrap.getBoundingClientRect();
    let ttx = (ev.clientX - wrapRect.left) + 14;
    let tty = (ev.clientY - wrapRect.top) + 14;
    tt.style.left = ttx + 'px'; tt.style.top = tty + 'px';
  });
  overlay.addEventListener('mouseleave', () => { hover.setAttribute('opacity',0); tt.style.opacity = 0; });
}

function legendHTML(items){
  return items.map(it => '<span><span class="sw" style="background:'+it.color+'"></span>'+it.label+'</span>').join('');
}

// ---- Chart 1: training loss ----
drawChart('svg-loss', {
  width:600, height:360, margin:{top:24,right:20,bottom:40,left:44},
  xDomain:[1,18], yDomain:[0.20,0.29],
  xTicks:[1,4,8,12,16,18], yTicks:[0.20,0.22,0.24,0.26,0.28],
  yFormat: v=>v.toFixed(2), yAxisLabel:'flow-matching loss',
  series:[
    {label:'W5 (pos_weight=5)', color:'var(--w5)', points:DATA.w5_flow_loss, raw:true, mean:true, dots:false},
    {label:'W1 (pos_weight=1)', color:'var(--w1)', points:DATA.w1_flow_loss, raw:true, mean:true, dots:false},
  ]
});
document.getElementById('legend-loss').innerHTML = legendHTML([
  {label:'W5 &mdash; faint=raw, bold=3-ep mean', color:'var(--w5)'},
  {label:'W1 &mdash; faint=raw, bold=3-ep mean', color:'var(--w1)'},
]);

// ---- Chart 2: quick-val IoU ----
drawChart('svg-iou', {
  width:600, height:360, margin:{top:24,right:20,bottom:40,left:40},
  xDomain:[2,18], yDomain:[0,0.26],
  xTicks:[2,4,6,8,10,12,14,16,18], yTicks:[0,0.05,0.10,0.15,0.20,0.25],
  yFormat: v=>v.toFixed(2), yAxisLabel:'IoU (16-sample quick-val)',
  refLines:[{y:0.235, label:'0.235 zero-shot oracle'}, {y:0.230, label:'0.230 prior best fine-tune'}],
  refLabelAnchor:'start',
  series:[
    {label:'W5', color:'var(--w5)', points:DATA.w5_val_iou, markBest:true, bestX:14},
    {label:'W1', color:'var(--w1)', points:DATA.w1_val_iou, markBest:true, bestX:16},
  ]
});
document.getElementById('legend-iou').innerHTML = legendHTML([
  {label:'W5 (best 0.215 @ep14)', color:'var(--w5)'},
  {label:'W1 (best 0.221 @ep16)', color:'var(--w1)'},
]) + '<span><span class="ln"></span>reference thresholds</span>';

// ---- Chart 3: overfit-1 controls ----
drawChart('svg-of1', {
  width:600, height:260, margin:{top:20,right:20,bottom:36,left:38},
  xDomain:[5,30], yDomain:[0,1.0],
  xTicks:[5,10,15,20,25,30], yTicks:[0,0.25,0.5,0.75,1.0],
  yFormat: v=>v.toFixed(2), yAxisLabel:'IoU (1-sample quick-val)',
  series:[
    {label:'W5 control', color:'var(--w5)', points:DATA.overfit1_w5_iou, markBest:true, bestX:30},
    {label:'W1 control', color:'var(--w1)', points:DATA.overfit1_w1_iou, markBest:true, bestX:30},
  ]
});
document.getElementById('legend-of1').innerHTML = legendHTML([
  {label:'W5 (0.885 @ep30)', color:'var(--w5)'},
  {label:'W1 (0.918 @ep30)', color:'var(--w1)'},
]);

// ---- Chart 4: 10-sample gates ----
const of10combined = DATA.overfit10_gate_iou.concat(DATA.overfit10_ext_iou);
drawChart('svg-of10', {
  width:600, height:260, margin:{top:20,right:20,bottom:36,left:38},
  xDomain:[5,300], yDomain:[0,0.6],
  xTicks:[5,50,100,150,200,250,300], yTicks:[0,0.15,0.3,0.45,0.6],
  yFormat: v=>v.toFixed(2), yAxisLabel:'IoU (10-sample quick-val)',
  series:[
    {label:'canon_overfit10 gate→ext', color:'var(--aux1)', points:of10combined, markBest:true, bestX:270},
    {label:'amended gate (train_1k_gate10)', color:'var(--aux2)', points:DATA.amended_gate_iou, markBest:true, bestX:150},
  ]
});
document.getElementById('legend-of10').innerHTML = legendHTML([
  {label:'canon_overfit10 gate→ext, 300ep (peak 0.476@ep230)', color:'var(--aux1)'},
  {label:'amended gate, 150ep (0.554@ep150, still climbing)', color:'var(--aux2)'},
]);
</script>
"""


# ==================================================================== Overview
overview_body = """
    <div class="callout">
      <strong>What ran:</strong> two 18-epoch fine-tunes of SegviGen's <code>full_seg</code> model
      on 1,123 real training shapes (<code>train_1k</code>, real DINOv3 photo conditioning),
      warm-started from the pretrained <code>full_seg.ckpt</code>. The only difference between the
      two arms is the loss weight on emissive voxels: <span class="tag w5">W5</span> uses
      5&times; weight ("eager"), <span class="tag w1">W1</span> uses plain 1&times; weight
      ("timid", the control). Both ran on the same L40S GPU class, same data, same schedule.
    </div>
    <div class="stat-row">
      <div class="stat"><b>231171 / 231172</b><span>Slurm job IDs (W5 / W1)</span></div>
      <div class="stat"><b>06:28 / 06:27</b><span>wall-clock (h:mm), both COMPLETED</span></div>
      <div class="stat"><b>18 &times; 1,123</b><span>epochs &times; training samples</span></div>
      <div class="stat"><b>full_seg.ckpt</b><span>warm-start, real DINOv3 cond</span></div>
    </div>
"""

# ==================================================================== Curves
run_config_rows = """
          <tr><td>Job ID</td><td class="num">231171</td><td class="num">231172</td><td><code>sacct</code></td></tr>
          <tr><td>Node</td><td class="num">cs-venus-15</td><td class="num">cs-venus-15</td><td><code>sacct</code> NodeList</td></tr>
          <tr><td>GPU</td><td class="num">1&times; L40S</td><td class="num">1&times; L40S</td><td>sbatch <code>--gres=gpu:l40s:1</code>; <code>sacct</code> confirms gres/gpu=1 (type not in accounting)</td></tr>
          <tr><td>Wall-clock</td><td class="num">06:28:02</td><td class="num">06:27:06</td><td><code>sacct</code> Elapsed</td></tr>
          <tr><td>Dataset</td><td class="num">train_1k (1,123)</td><td class="num">train_1k (1,123)</td><td>log <code>[data]</code> line</td></tr>
          <tr><td>Epochs</td><td class="num">18</td><td class="num">18</td><td>SubmitLine arg 1</td></tr>
          <tr><td>Steps / epoch</td><td class="num">1,123</td><td class="num">1,123</td><td><code>n_per_epoch=0</code> &rarr; full dataset each epoch</td></tr>
          <tr><td>Batch size</td><td class="num">1 (no grad accum)</td><td class="num">1 (no grad accum)</td><td>train_emissive.py: per-sample loop, <code>opt.step()</code> every sample</td></tr>
          <tr><td>Total optimizer steps</td><td class="num">20,214</td><td class="num">20,214</td><td>18 &times; 1,123</td></tr>
          <tr><td>Optimizer</td><td class="num">AdamW, wd=0.0</td><td class="num">AdamW, wd=0.0</td><td><code>torch.optim.AdamW(...)</code></td></tr>
          <tr><td>Learning rate</td><td class="num">1e-5, constant</td><td class="num">1e-5, constant</td><td>sbatch <code>--lr 1e-5</code>; no scheduler in code</td></tr>
          <tr><td>Grad clip</td><td class="num">max-norm 1.0</td><td class="num">max-norm 1.0</td><td><code>clip_grad_norm_(..., 1.0)</code></td></tr>
          <tr><td>Gradient checkpointing</td><td class="num">on, 31 modules</td><td class="num">on, 31 modules</td><td>log <code>[mem]</code> line, both jobs</td></tr>
          <tr><td>Precision</td><td class="num">bf16</td><td class="num">bf16</td><td>model variant <code>..._dit_1_3B_512_bf16</code>; no autocast wrapper &mdash; trains in the checkpoint's native dtype</td></tr>
          <tr><td>Init checkpoint</td><td class="num">full_seg.ckpt</td><td class="num">full_seg.ckpt</td><td>log <code>[init]</code> line (SegviGen HF snapshot)</td></tr>
          <tr><td>Cond mode</td><td class="num">real (DINOv3)</td><td class="num">real (DINOv3)</td><td>SubmitLine arg 4 / log <code>[data]</code> line</td></tr>
          <tr><td>Oversampling</td><td class="num">on, pow=1.0</td><td class="num">on, pow=1.0</td><td><code>--emis_oversample</code> flag (sbatch); <code>oversample_pow</code> default 1.0 (not overridden)</td></tr>
          <tr><td>pos_weight</td><td class="num">5.0</td><td class="num">1.0</td><td>SubmitLine arg 5</td></tr>
          <tr><td>EMA decay</td><td class="num">0.999</td><td class="num">0.999</td><td>log <code>[ema]</code> line; argparse default, not overridden</td></tr>
          <tr><td>Save / quick-val cadence</td><td class="num">every 2 ep, 16-sample</td><td class="num">every 2 ep, 16-sample</td><td>SubmitLine args 7&ndash;8 (<code>save_every=2</code>, <code>val_quick=16</code> on val_96)</td></tr>
"""

curves_body = f"""
    <p class="sub">The two arms that matter: <span class="tag w5">W5</span> pos_weight=5.0 vs
      <span class="tag w1">W1</span> pos_weight=1.0, otherwise identical. DiffusionNet is not shown
      anywhere on this page &mdash; that baseline was abandoned earlier in the project.</p>

    <div id="curve-charts">
    <div class="panel-grid">
      <div class="chart-card">
        <h3>Training loss (flow-matching) <span class="hint">per epoch, both arms</span></h3>
        <div class="diagram-wrap wide-diagram" id="wrap-loss">
          <svg viewBox="0 0 600 360" id="svg-loss"></svg>
        </div>
        <div class="legend" id="legend-loss"></div>
        <p class="caption">Faint line = raw per-epoch loss; bold line = 3-epoch running mean. Flow
          loss is inherently noisy (it's a per-batch denoising objective, not a monotonic metric) &mdash;
          the running mean is there to separate real trend from batch noise.</p>
      </div>
      <div class="chart-card">
        <h3>Quick-val IoU <span class="hint">16-sample val_96 subset, every 2 epochs</span></h3>
        <div class="diagram-wrap wide-diagram" id="wrap-iou">
          <svg viewBox="0 0 600 360" id="svg-iou"></svg>
        </div>
        <div class="legend" id="legend-iou"></div>
        <p class="caption">Reference lines: <strong>0.235</strong> = zero-shot SegviGen
          segment-then-label oracle (upper bound on this task); <strong>0.230</strong> = best
          fine-tune result before Phase 4 (real-cond, job 226802, ep2). Best epochs are marked.
          This is a 16-sample estimate &mdash; expect &plusmn;0.03&ndash;0.05 noise from sampling alone,
          which is most of why the curve isn't smooth.</p>
      </div>
    </div>
    </div>

    <p>Two different things are being plotted and they don't move together.
      <strong>Training loss</strong> (left chart above) is a flow-matching denoising loss &mdash;
      it tells you how well the model predicts a training-time noise sample, averaged over an
      epoch. It is expected to fall fast in the first few epochs and then plateau into a noisy
      band; it is <em>not</em> the metric anyone cares about and can look flat while the model is
      still improving on the metric that matters.</p>
    <p><strong>IoU</strong> (right chart above, and both charts in Gates below) is the metric
      that matters: it's computed by actually running the diffusion sampler to generate a
      prediction, then measuring overlap with ground truth. Sampling is stochastic, and on top of
      that the quick-val numbers use only 16 shapes &mdash; so a single epoch's IoU is a noisy estimate,
      not a precise measurement. That's why the IoU curves bounce around by &plusmn;0.03&ndash;0.05
      between adjacent epochs even when nothing about training has changed, and why "best epoch"
      is read off a peak in a noisy curve rather than a clean monotonic climb.</p>

    <div id="run-config">
    <h3 style="margin-top:1.6rem">Run configuration</h3>
    <p class="sub">Pulled from <code>sacct</code>, the sbatch <code>SubmitLine</code>,
      {lp.filepath("train_emissive_v4.sbatch", f"{SEGVIGEN_ROOT}/code/train_emissive_v4.sbatch")}, and
      {lp.filepath("train_emissive.py", f"{SEGVIGEN_ROOT}/code/train_emissive.py")}'s argparse
      defaults / training loop &mdash; not from memory. Where the same value applies to both
      arms it's simply repeated in both columns.</p>
    <div class="table-scroll">
      <table class="results">
        <thead><tr><th>Setting</th><th><span class="tag w5">W5</span></th><th><span class="tag w1">W1</span></th><th>Source</th></tr></thead>
        <tbody>{run_config_rows}
        </tbody>
      </table>
    </div>
    <p class="caption">GPU memory class: the training script's own comment states gradient
      checkpointing "lets the full fine-tune fit on a 44GB GPU (l40s/a40)" &mdash; quoted directly
      from {lp.filepath("train_emissive.py", f"{SEGVIGEN_ROOT}/code/train_emissive.py")} rather than asserted from general L40S specs.</p>
    </div>
"""

# ==================================================================== Gates
gates_body = """
    <p class="sub">Before committing 6.5 GPU-hours per arm to the full 1k-sample run, every change
      was sanity-gated on tiny subsets first: can the trainer overfit one shape? Ten shapes? These
      four earlier jobs are why Phase&nbsp;4 was trusted enough to launch.</p>

    <div class="panel-grid">
      <div class="chart-card">
        <h3>Overfit-1 controls <span class="hint">jobs 231118 (W5) / 231119 (W1), 1 sample</span></h3>
        <div class="diagram-wrap wide-diagram" id="wrap-of1">
          <svg viewBox="0 0 600 260" id="svg-of1"></svg>
        </div>
        <div class="legend" id="legend-of1"></div>
        <p class="caption">Proof the trainer can fit at all: both arms climb to near-perfect
          single-shape IoU by epoch 30 (W5 0.885, W1 0.918). Neither collapses or diverges.</p>
      </div>
      <div class="chart-card">
        <h3>10-sample gates <span class="hint">jobs 231120+231124 (canon_overfit10) &amp; 231164 (amended)</span></h3>
        <div class="diagram-wrap wide-diagram" id="wrap-of10">
          <svg viewBox="0 0 600 260" id="svg-of10"></svg>
        </div>
        <div class="legend" id="legend-of10"></div>
        <p class="caption">canon_overfit10 gate (231120, 40ep) continues into a 260-epoch extension
          (231124, warm-started from its own epoch 40) &mdash; slow, noisy climb, peak 0.476@ep230. The
          amended gate (231164, a different 10-shape subset, <code>train_1k_gate10</code>) climbs
          more cleanly to 0.554 at epoch 150 and was still rising when the run ended.</p>
      </div>
    </div>
"""

# ==================================================================== Final results
final_results_body = f"""
    <p class="sub">Each Phase 4 checkpoint's <code>best.ckpt</code> and EMA shadow weights were
      evaluated once each on the full 96-shape held-out set ({lp.filepath("eval_val96.sbatch", f"{SEGVIGEN_ROOT}/code/eval_val96.sbatch")},
      fixed-threshold sweep + Otsu). "Flat mean" below is the best of the four fixed thresholds
      {{0.2, 0.3, 0.4, 0.5}} for that checkpoint.</p>
    <div class="table-scroll">
      <table class="results">
        <thead>
          <tr><th>Checkpoint</th><th class="num">Flat mean IoU</th><th>Best threshold</th><th>Source log</th></tr>
        </thead>
        <tbody>
          <tr><td><span class="tag w5">W5</span> best (epoch 14)</td><td class="num">0.117</td><td>@0.2</td><td>{lp.filepath("eval96_231256.log", f"{SEGVIGEN_ROOT}/eval96_231256.log")}</td></tr>
          <tr><td><span class="tag w1">W1</span> best (epoch 16)</td><td class="num">0.161</td><td>@0.5</td><td>{lp.filepath("eval96_231257.log", f"{SEGVIGEN_ROOT}/eval96_231257.log")}</td></tr>
          <tr><td><span class="tag w5">W5</span> EMA (epoch 14)</td><td class="num">0.096</td><td>@0.2</td><td>{lp.filepath("eval96_231258.log", f"{SEGVIGEN_ROOT}/eval96_231258.log")}</td></tr>
          <tr><td><span class="tag w1">W1</span> EMA (epoch 16)</td><td class="num">0.172</td><td>@0.3</td><td>{lp.filepath("eval96_231259.log", f"{SEGVIGEN_ROOT}/eval96_231259.log")}</td></tr>
        </tbody>
      </table>
    </div>
    <div class="note">
      <strong>Honest flag, not silently smoothed over:</strong> the two EMA checkpoints were
      re-evaluated a day later for the visual-predictions page ({lp.filepath("eval96_231379.log", f"{SEGVIGEN_ROOT}/eval96_231379.log")} /
      {lp.filepath("eval96_231380.log", f"{SEGVIGEN_ROOT}/eval96_231380.log")}, identical checkpoint files) and came back noticeably higher
      &mdash; W5-EMA 0.096 &rarr; 0.128, W1-EMA 0.172 &rarr; up to 0.193. Same weights, same eval
      script, different result, because generation is sampled stochastically. The table above
      keeps the original numbers this project has been quoting; the rerun is a live demonstration
      of exactly the sampling noise described in the Curves section, not a correction.
    </div>
    <p style="margin-top:.9rem">For the actual predicted-vs-ground-truth renders on 8 real
      val_96 shapes, see
      <a href="../finetune_binary_v1/index.html">finetune_binary_v1</a>. This page covers
      the original 1k-shape fine-tune only &mdash; for the follow-up 2k-shape run (bigger data
      + balanced loss weighting, still below the zero-shot oracle), see
      <a href="../results_2k_v1/index.html">latest 2k results &rarr;</a>.</p>
"""

body_sections = [
    lp.section("overview", 1, "Overview", body_html=overview_body, preview_rem=None),
    lp.section("curves", 2, "Phase 4 &mdash; the fine-tuning runs",
               takeaway="Both arms train stably for all 18 epochs &mdash; but on the metric that "
                        "matters (quick-val IoU) neither beats the 0.235 zero-shot oracle, and much "
                        "of the epoch-to-epoch bounce is 16-sample evaluation noise, not real signal.",
               body_html=curves_body, preview_rem=53),
    lp.section("gates", 3, "The gates that came before",
               takeaway="Four earlier sanity-gate jobs (1-sample and 10-sample overfit tests) proved "
                        "the trainer could fit at all before 6.5 GPU-hours per arm went into the full "
                        "1k-sample run &mdash; both overfit-1 controls reach IoU &gt;0.88 by epoch 30.",
               body_html=gates_body, preview_rem=48.5),
    lp.section("final-results", 4, "Full val_96 result (96 shapes, not the 16-sample quick-val)",
               takeaway="Flat-mean IoU: W5 0.117 / W1 0.161 / W5-EMA 0.096 / W1-EMA 0.172 &mdash; all "
                        "below the 0.230/0.235 reference lines, and a same-checkpoint rerun a day later "
                        "moved these by up to 0.02, which is the point about evaluation noise made "
                        "concrete.",
               body_html=final_results_body, preview_rem=None),
    ('<footer>\n'
     '    Parsed from ' +
     lp.filepath("train_curve.json", f"{SEGVIGEN_ROOT}/outputs/emis_1k_w5/train_curve.json") + ' (W5) / ' +
     lp.filepath("train_curve.json", f"{SEGVIGEN_ROOT}/outputs/emis_1k_w1/train_curve.json") + ' (W1), ' +
     lp.filepath("log.json", f"{SEGVIGEN_ROOT}/outputs/emis_1k_w5/log.json") + ' (W5) / ' +
     lp.filepath("log.json", f"{SEGVIGEN_ROOT}/outputs/emis_1k_w1/log.json") + ' (W1),\n'
     '    ' + ", ".join(
         lp.filepath(f"train_{j}.log", f"{SEGVIGEN_ROOT}/train_{j}.log")
         for j in ["231118", "231119", "231120", "231124", "231164", "231171", "231172"]
     ) + ' and\n'
     '    ' + ", ".join(
         lp.filepath(f"eval96_{j}.log", f"{SEGVIGEN_ROOT}/eval96_{j}.log")
         for j in ["231256", "231257", "231258", "231259", "231379", "231380"]
     ) + ' on\n'
     '    ' + lp.filepath(f"{SEGVIGEN_ROOT}/", SEGVIGEN_ROOT + "/") + ' &middot;\n'
     '    generated 2026-07-03, ported to xgpage.py 2026-07-06 &middot;\n'
     '    <a href="../index.html">&uarr; all lightgen visuals</a>\n'
     '  </footer>'),
]

html = lp.page(
    title="SegviGen emissive fine-tune — training curves",
    header_html=lp.header(
        "Did the fine-tuning really happen?",
        'Yes &mdash; here is every loss and validation curve from the two Phase&nbsp;4 training '
        'jobs, plus the sanity-gate runs that preceded them, parsed straight from the Slurm logs and '
        '<code>train_curve.json</code> checkpoints on disk. No cherry-picking: every point below is '
        'what the logs say, including the epochs where both arms got worse.'),
    body_sections=body_sections,
    outline_entries=[
        {"id": "overview", "label": "Overview"},
        {"id": "curves", "label": "Curves", "sub": [
            {"id": "curve-charts", "label": "Loss + IoU charts"},
            {"id": "run-config", "label": "Run configuration"},
        ]},
        {"id": "gates", "label": "Gates"},
        {"id": "final-results", "label": "Final results"},
    ],
    needs_katex=False,
    extra_head=f"<style>{PAGE_STYLE}</style>\n",
    extra_body_end=CHART_SCRIPT,
)

with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print(f"\nwrote {OUT}/index.html")
