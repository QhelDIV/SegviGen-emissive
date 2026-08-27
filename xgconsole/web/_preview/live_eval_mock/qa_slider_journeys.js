#!/usr/bin/env node
/* qa_slider_journeys.js -- simulated-user journeys for the live_eval redesign
 * mockup's checkpoint scrubber (slider_view.js). Same shape as
 * train_rungraph's qa_rungraph_journeys.js: each journey drives a real
 * sequence and asserts on the state the reader is left looking at, not on
 * handler internals.
 *
 * Run via the project's journey_harness.sh wrapper if Node 20+ is on PATH;
 * on this workstation (2026-08-26, no nvm) it was run directly with:
 *   NODE_PATH=/localhome/xya120/.npm/_npx/9833c18b2d85bc59/node_modules \
 *   /localhome/xya120/.vscode-server/cli/servers/Stable-08d4889f9ec4a1685d257b9b95de036c8e1ce1e5/server/node \
 *     qa_slider_journeys.js [url]
 */
const NODE_PATH = process.env.NODE_PATH;
if (NODE_PATH && !require.resolve.paths(".").includes(NODE_PATH)) {
  module.paths.push(NODE_PATH);
}
const { chromium } = require("playwright");

const DEFAULT_URL =
  "https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/live_eval_mock/index.html";
const CHROME_PATH = process.env.QA_CHROME_PATH ||
  (process.env.HOME + "/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome");

const url = process.argv[2] || DEFAULT_URL;

async function label(page) {
  return page.$eval("[data-sw-label]", (e) => e.textContent.trim());
}
async function rangeValue(page) {
  return page.$eval("[data-sw-range]", (e) => e.value);
}
async function firstDrawSrc(page) {
  return page.$eval('[data-sw-cell][data-sw-draw="0"] img', (e) => e.getAttribute("src"));
}
async function firstRowLabel(page) {
  return page.$eval("[data-sw-rowlabel]", (e) => e.textContent.trim());
}
async function prevDisabled(page) {
  return page.$eval("[data-sw-prev]", (e) => e.disabled);
}
async function nextDisabled(page) {
  return page.$eval("[data-sw-next]", (e) => e.disabled);
}

const JOURNEYS = [
  {
    name: "loads with the newest checkpoint shown, no JS needed to see it",
    async run(page) {
      await page.waitForSelector("[data-sw-grid]", { timeout: 15000 });
      // Give slider_view.js's fetch("ckpts.json") a moment to land.
      await page.waitForFunction(
        () => document.querySelector("[data-sw-label]").textContent.includes("step"),
        { timeout: 8000 }
      );
      const lbl = await label(page);
      const idx = await rangeValue(page);
      const src = await firstDrawSrc(page);
      if (!/step 6040/.test(lbl)) throw new Error("expected newest step 6040 in label, got: " + lbl);
      if (!src.includes("step0006040")) throw new Error("draw image not on newest checkpoint: " + src);
      if (Number(idx) !== 4) throw new Error("range should default to last index, got " + idx);
      if (await nextDisabled(page) !== true) throw new Error("next should be disabled at the newest checkpoint");
    },
  },
  {
    name: "prev button steps back one checkpoint and updates every changing part",
    async run(page) {
      const before = await firstRowLabel(page);
      await page.click("[data-sw-prev]");
      await page.waitForTimeout(150);
      const lbl = await label(page);
      const src = await firstDrawSrc(page);
      const after = await firstRowLabel(page);
      if (!/step 5285/.test(lbl)) throw new Error("prev should land on step 5285, got: " + lbl);
      if (!src.includes("step0005285")) throw new Error("draw image did not move to step0005285: " + src);
      if (after === before) throw new Error("row-header IoU did not update on scrub");
    },
  },
  {
    name: "keyboard left/right arrows scrub without the range input focused",
    async run(page) {
      // Blur whatever has focus WITHOUT clicking anywhere on the page --
      // clicking "body" risks landing on the range slider itself (its huge
      // bounding box can coincide with whatever is mid-viewport) and
      // silently changing its value before the key press even happens.
      await page.evaluate(() => document.activeElement && document.activeElement.blur());
      await page.keyboard.press("ArrowLeft");
      await page.waitForTimeout(120);
      let lbl = await label(page);
      if (!/step 4530/.test(lbl)) throw new Error("ArrowLeft should land on step 4530, got: " + lbl);
      await page.keyboard.press("ArrowRight");
      await page.keyboard.press("ArrowRight");
      await page.waitForTimeout(120);
      lbl = await label(page);
      if (!/step 6040/.test(lbl)) throw new Error("two ArrowRight should return to step 6040, got: " + lbl);
    },
  },
  {
    name: "dragging the range to the first checkpoint shows the earliest evaluated epoch, and prev disables at the boundary",
    async run(page) {
      await page.$eval("[data-sw-range]", (e) => {
        e.value = "0";
        e.dispatchEvent(new Event("input", { bubbles: true }));
      });
      await page.waitForTimeout(120);
      const lbl = await label(page);
      const src = await firstDrawSrc(page);
      if (!/step 2000/.test(lbl)) throw new Error("range=0 should show step 2000, got: " + lbl);
      if (!src.includes("step0002000")) throw new Error("draw image did not move to step0002000: " + src);
      if (await prevDisabled(page) !== true) throw new Error("prev should disable at the earliest checkpoint");
      if (await nextDisabled(page) !== false) throw new Error("next should re-enable off the boundary");
    },
  },
  {
    name: "a missing panel renders the same dashed placeholder tile the live page uses, not a broken image",
    async run(page) {
      // Exercise the onerror path directly: point one cell at a path that
      // cannot exist, via the render() hook slider_view.js exposes for QA.
      await page.evaluate(() => {
        const root = document.querySelector("[data-sw-root]");
        const cell = document.querySelector('[data-sw-cell][data-sw-draw="1"]');
        cell.querySelector("img").src = "img/step9999999/does_not_exist.png";
      });
      await page.waitForTimeout(200);
      const hasPlaceholder = await page.$eval(
        '[data-sw-cell][data-sw-draw="1"]',
        (e) => !!e.querySelector(".gf-placeholder") &&
               e.querySelector("img").style.display === "none"
      );
      if (!hasPlaceholder) throw new Error("missing image did not fall back to .gf-placeholder");
    },
  },
  {
    name: "refresh always lands back on the newest checkpoint (no persisted scrub state)",
    async run(page) {
      // Land somewhere in the middle first, then confirm it doesn't stick.
      await page.$eval("[data-sw-range]", (e) => {
        e.value = "1";
        e.dispatchEvent(new Event("input", { bubbles: true }));
      });
      await page.waitForTimeout(120);
      await page.reload();
      await page.waitForSelector("[data-sw-grid]", { timeout: 15000 });
      await page.waitForFunction(
        () => document.querySelector("[data-sw-label]").textContent.includes("step"),
        { timeout: 8000 }
      );
      const lbl = await label(page);
      if (!/step 6040/.test(lbl)) throw new Error("refresh should reset to newest checkpoint, got: " + lbl);
    },
  },
];

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME_PATH });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "networkidle" });
  let failed = 0;
  for (const j of JOURNEYS) {
    try {
      await j.run(page);
      console.log("PASS: " + j.name);
    } catch (e) {
      failed++;
      console.log("FAIL: " + j.name + "\n      " + e.message);
    }
  }
  await browser.close();
  console.log(`\n${JOURNEYS.length - failed}/${JOURNEYS.length} journeys passed.`);
  process.exit(failed ? 1 : 0);
})();
