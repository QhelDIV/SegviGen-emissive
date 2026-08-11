#!/usr/bin/env node
/* qa_graph_journeys.js — the page-relationship graph's simulated-user QA
 * journeys (permanent deliverable, 2026-08-10, owner-directed round 2).
 *
 * WHY THIS EXISTS: the owner's round-1 complaint ("if I click one node,
 * things get cluttered, and I have to refresh the page to reset") was a
 * real bug (click used to open a new tab immediately; a hover-then-click
 * sequence could leave the original tab's dim/highlight permanently stuck,
 * because opening a new tab does not reliably fire pointerleave on the page
 * left behind). A handler-level test ("did click fire") would never have
 * caught this -- the handler DID fire, correctly, every time. What was
 * missing was a test that DRIVES a realistic multi-step sequence and
 * asserts on the RESULTING VISIBLE STATE afterward. This script is that:
 * it is the gate for any future change to web/assets/graph_view.js's
 * interaction code, not a one-off test run. Re-run it after any edit to
 * the click/hover/search/select state machine, before publishing.
 *
 * EXTRACTED 2026-08-10 (xgpage.graph's package extraction, journey-harness
 * round): the generic runner loop (browser lifecycle, per-journey error
 * handling, PASS/FAIL reporting, exit code) now lives in the package as
 * xgpage's bundled qa/journey_harness.js, alongside qa_widths.js and
 * qa_v3_interact.js (see xgpage.publish.qa_path()). This file keeps only
 * what is genuinely specific to the graph page: the DOM helpers that know
 * ".gn-node"/"#graph-svg"/"#graph-search" (nodeBox, bgPoint, snapshot,
 * isBaseline) and the seven JOURNEYS themselves. A project with a
 * DIFFERENT interactive page copies this file's shape, not its content.
 *
 * USAGE, the way anyone on this workstation should actually run it:
 *   tools/qa_graph_journeys.sh [url]
 * That wrapper finds a working Node 20+ (Playwright's minimum; the default
 * /usr/bin/node here is v12) and the isolated Playwright install, and fails
 * with a clear message if it can't -- see its own header for exactly what
 * it checks and the QA_NODE_BIN / QA_NODE_PATH overrides. Found live
 * (2026-08-10): running this .js file directly with plain `node` (no
 * NODE_PATH, whatever node happens to be on PATH) is exactly what a fresh
 * agent will try first, and on this machine that fails with a cryptic
 * MODULE_NOT_FOUND for "playwright" -- the wrapper exists so nobody has to
 * rediscover the working invocation from scratch. If you need to invoke
 * this file directly (e.g. from another script that already resolved its
 * own node/env), the equivalent manual form is:
 *   NODE_PATH=<dir containing an installed playwright package> \
 *     <a node >=20 binary> tools/qa_graph_journeys.js [url]
 *   (url defaults to the live console graph page; pass a staging URL to
 *   test before publishing.)
 *
 * Exits non-zero if any journey fails, so it can gate a build/publish step.
 *
 * MODEL UNDER TEST (see graph_view.js's own "interaction state" comment
 * block for the implementation): ui.hoverId (transient, cleared on
 * pointerleave / window blur / visibilitychange), ui.selectedId (sticky,
 * set by a plain click, toggled off by clicking the same node again),
 * ui.searchHits (the search box's current matches). ONE reset path clears
 * all three at once: a background click (tap with no pan movement) or the
 * Escape key. A double-click opens the page in a new tab and does not
 * itself change ui state beyond defensively clearing hover.
 */
"use strict";
// The package's installed location on this workstation (canonical xgpage
// checkout, see the xgpage skill) -- XGPAGE_QA_DIR overrides for a
// different machine/checkout, same override pattern as QA_CHROME_PATH
// below.
const QA_DIR = process.env.XGPAGE_QA_DIR || (process.env.HOME + "/studio/xgpage/src/xgpage/qa");
const { runJourneys } = require(QA_DIR + "/journey_harness.js");

const DEFAULT_URL = "https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/graph.html";
const CHROME_PATH = process.env.QA_CHROME_PATH ||
  (process.env.HOME + "/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome");

async function nodeBox(page, id) {
  return page.$eval('.gn-node[data-id="' + id + '"] .gn-circle', (el) => {
    const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
}

async function bgPoint(page, corner) {
  return page.$eval("#graph-svg", (el, c) => {
    const r = el.getBoundingClientRect();
    return c === "br" ? { x: r.x + r.width - 20, y: r.y + r.height - 20 } : { x: r.x + 20, y: r.y + 20 };
  }, corner);
}

async function snapshot(page) {
  return page.evaluate(() => {
    const svg = document.querySelector("#graph-svg");
    return {
      dimmed: svg.classList.contains("dimmed"),
      hi: [...document.querySelectorAll(".gn-node.hi")].map((e) => e.dataset.id).sort(),
      selected: [...document.querySelectorAll(".gn-node.selected")].map((e) => e.dataset.id).sort(),
      searchHit: [...document.querySelectorAll(".gn-node.search-hit")].map((e) => e.dataset.id).sort(),
      searchVal: document.querySelector("#graph-search").value,
    };
  });
}

function isBaseline(s) {
  return !s.dimmed && s.hi.length === 0 && s.selected.length === 0 && s.searchHit.length === 0 && s.searchVal === "";
}

function eqArr(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

// ------------------------------------------------------------- journeys --
// Each journey returns {name, pass, detail}. All operate on a fresh page
// (see main()) so journeys never interfere with each other.
const JOURNEYS = [

  {
    name: "J1 hover is transient (leaves no trace once the pointer moves away)",
    async run(page) {
      const box = await nodeBox(page, "training_curves_v1");
      await page.mouse.move(box.x, box.y);
      await page.waitForTimeout(150);
      const during = await snapshot(page);
      await page.mouse.move(10, 10);
      await page.waitForTimeout(150);
      const after = await snapshot(page);
      const pass = during.dimmed === true && during.hi.includes("training_curves_v1") && isBaseline(after);
      return { pass, detail: { during, after } };
    },
  },

  {
    name: "J2 click selects (dim + neighborhood emphasis); background click deselects to baseline",
    async run(page) {
      const box = await nodeBox(page, "training_curves_v1");
      await page.mouse.click(box.x, box.y);
      await page.waitForTimeout(150);
      const during = await snapshot(page);
      const bg = await bgPoint(page, "tl");
      await page.mouse.click(bg.x, bg.y);
      await page.waitForTimeout(150);
      const after = await snapshot(page);
      const pass = eqArr(during.selected, ["training_curves_v1"]) && during.dimmed === true && isBaseline(after);
      return { pass, detail: { during, after } };
    },
  },

  {
    name: "J3 click selects; Escape deselects to baseline",
    async run(page) {
      const box = await nodeBox(page, "training_curves_v1");
      await page.mouse.click(box.x, box.y);
      await page.waitForTimeout(150);
      const during = await snapshot(page);
      await page.keyboard.press("Escape");
      await page.waitForTimeout(150);
      const after = await snapshot(page);
      const pass = eqArr(during.selected, ["training_curves_v1"]) && isBaseline(after);
      return { pass, detail: { during, after } };
    },
  },

  {
    name: "J4 search highlights matches; Escape clears the highlight AND the search box value",
    async run(page) {
      await page.fill("#graph-search", "training");
      await page.waitForTimeout(300);
      const during = await snapshot(page);
      await page.keyboard.press("Escape");
      await page.waitForTimeout(150);
      const after = await snapshot(page);
      const pass = during.searchHit.length > 0 && during.searchVal === "training" && isBaseline(after);
      return { pass, detail: { during, after } };
    },
  },

  {
    name: "J5 the owner's original bug: hover then double-click (opens a new tab) never strands the original tab",
    async run(page, ctx) {
      const box = await nodeBox(page, "training_curves_v1");
      await page.mouse.move(box.x, box.y);
      await page.waitForTimeout(100);
      const [popup] = await Promise.all([
        ctx.waitForEvent("page"),
        page.mouse.dblclick(box.x, box.y),
      ]);
      await popup.waitForLoadState();
      const popupUrl = popup.url();
      await popup.close();
      await page.waitForTimeout(250);
      const after = await snapshot(page);
      // "not stranded" means: no dimmed class stuck on with no selection to
      // justify it, and nothing selected either (a double-click's two single
      // clicks toggle selection on then off -- see graph_view.js comment).
      const pass = !after.dimmed && after.selected.length === 0 && popupUrl.includes("training_curves_v1");
      return { pass, detail: { after, popupUrl } };
    },
  },

  {
    name: "J6 the owner's exact clutter sequence, chained: hover, click, hover, click, drag, search, Enter, Escape -> baseline",
    async run(page) {
      let a = await nodeBox(page, "training_curves_v1");
      await page.mouse.move(a.x, a.y); await page.waitForTimeout(80);
      await page.mouse.click(a.x, a.y); await page.waitForTimeout(80);
      let b = await nodeBox(page, "results_2k_v1");
      await page.mouse.move(b.x, b.y); await page.waitForTimeout(80);
      await page.mouse.click(b.x, b.y); await page.waitForTimeout(80);
      let c = await nodeBox(page, "workspace");
      await page.mouse.move(c.x, c.y); await page.mouse.down();
      await page.mouse.move(c.x + 40, c.y + 30, { steps: 5 }); await page.waitForTimeout(80);
      await page.mouse.up(); await page.waitForTimeout(80);
      await page.fill("#graph-search", "render"); await page.waitForTimeout(200);
      await page.keyboard.press("Enter"); await page.waitForTimeout(80);
      await page.keyboard.press("Escape"); await page.waitForTimeout(150);
      const after = await snapshot(page);
      return { pass: isBaseline(after), detail: { after } };
    },
  },

  {
    name: "J7 drag never latches a mode: dragging a node changes only its position, not selection/dim state",
    async run(page) {
      const before = await snapshot(page);
      const c = await nodeBox(page, "workspace");
      await page.mouse.move(c.x, c.y);
      await page.mouse.down();
      await page.mouse.move(c.x + 60, c.y + 40, { steps: 8 });
      await page.waitForTimeout(150);
      const duringDrag = await snapshot(page);
      await page.mouse.up();
      await page.waitForTimeout(150);
      const after = await snapshot(page);
      // a drag SHOULD show transient hover/dim on the dragged node (the
      // pointer is resting on it) but must NEVER set .selected, and must
      // return to baseline once released with no further action.
      const pass = duringDrag.selected.length === 0 && isBaseline(before) && isBaseline(after);
      return { pass, detail: { before, duringDrag, after } };
    },
  },
];

async function main() {
  const url = process.argv[2] || DEFAULT_URL;
  const allPass = await runJourneys(url, JOURNEYS, { chromePath: CHROME_PATH, label: "graph interaction" });
  process.exit(allPass ? 0 : 1);
}

main();
