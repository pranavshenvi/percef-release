#!/usr/bin/env node
/*
 * Scorecard of every unified run against a tuned fixed ef at the same mean recall, recomputed from
 * the rows_{P,R}.json files alone (updateAsOf290926.md).
 *
 * "Tuned fixed ef at recall r" is interpolated between the two fixed-ef grid points around r in
 * log(1 - recall) vs log(cost) space. Linear interpolation in (recall, cost) overstates the fixed
 * ef's cost, because cost grows ever faster as recall approaches 1: in a leave-one-out test on the
 * fixed-ef grids it is +11% too high on average (at twice the grid spacing), log-log -0.4%. Both
 * are printed, with that test (--loo).
 *
 * Usage (from the repo root, Node >= 14, no packages):
 *   node analysis/rescore_scorecard.js [results_dir=server_results] [--loo] [--json out.json]
 */
const fs = require("fs"), path = require("path");
const argv = process.argv.slice(2);
const root = argv.find(a => !a.startsWith("--") && !a.endsWith(".json")) || "server_results";
const jsonOut = argv.includes("--json") ? argv[argv.indexOf("--json") + 1] : null;

// KS at 200 queries (survey_ks_vibe.py / survey_ks_standard.py; updateAsOf260926.md §6, 280926 §4)
const KS = {glove100: .018, dbpedia1536: .038, msmarco384: .044, cohere1024: .049, laion_i2i: .034,
  deepimage96: .066, yambda: .090, sift128: .126, vibe_landmark_dino: .076, vibe_inaturalist_resnet: .115,
  gist960: .091, fashionmnist784: .073, vibe_yahoo_minilm: .054, vibe_imagenet_align: .058,
  lastfm64: .216, deep1b: .067, coco_i2i: .046, coco_t2i: .056, bigann: .029, msturing: .010};
// queries from a different modality than the corpus: text->image, user->item
const CROSS_MODAL = new Set(["vibe_imagenet_align", "coco_t2i", "lastfm64"]);
const BAND_LO = 0.044, BAND_HI = 0.066;
const METHODS = {"Ada-ef (as shipped)": "Ada", "Ada-ef (WAE floor)": "AdaWAE", "Ours (K=1, Isotonic)": "Ours"};

const readJson = f => JSON.parse(fs.readFileSync(f, "utf8").replace(/\bNaN\b/g, "null"));
const X = r => Math.log(Math.max(1 - r, 1e-6));

function fixedAt(fixed, r, key, mode) {
  for (let i = 0; i + 1 < fixed.length; i++) {
    const a = fixed[i], c = fixed[i + 1];
    if (!(a.mean_r < r && r <= c.mean_r)) continue;
    if (a[key] == null || c[key] == null) return null;
    if (mode === "lin") return a[key] + (r - a.mean_r) / (c.mean_r - a.mean_r) * (c[key] - a[key]);
    const t = (X(r) - X(a.mean_r)) / (X(c.mean_r) - X(a.mean_r));
    if (key === "p1" || key === "p5") return a[key] + t * (c[key] - a[key]);
    return Math.exp(Math.log(a[key]) + t * Math.log(c[key] / a[key]));
  }
  return null;
}

function latestRuns() {
  const byDs = {};
  for (const d of fs.readdirSync(root).filter(d => d.startsWith("results_unified_") && !d.includes("smoke") && !d.includes("ablation") && !d.includes("_sweep"))) {
    const ds = d.replace("results_unified_", "").replace(/_\d{8}_\d{6}$/, "");
    if (!byDs[ds] || d > byDs[ds]) byDs[ds] = d;
  }
  return byDs;
}

const runs = latestRuns(), out = [], loo = {lin: [], log: []};
for (const [ds, dir] of Object.entries(runs)) {
  for (const S of ["P", "R"]) {
    const rf = path.join(root, dir, `rows_${S}.json`), sf = path.join(root, dir, `summary_${S}.json`);
    if (!fs.existsSync(rf)) continue;
    const rows = readJson(rf), sum = fs.existsSync(sf) ? readJson(sf) : {};
    const fixed = rows.filter(r => r.name.startsWith("Fixed")).sort((a, b) => a.avg_ef - b.avg_ef);
    for (let i = 1; S === "R" && i + 1 < fixed.length; i++) {   // leave-one-out test (fixed rows are shared by P and R)
      const [a, b, c] = [fixed[i - 1], fixed[i], fixed[i + 1]];
      if (b.mean_r > 0.995 || !(a.mean_r < b.mean_r && b.mean_r < c.mean_r)) continue;
      const pair = [a, c];
      for (const m of ["lin", "log"]) loo[m].push((fixedAt(pair, b.mean_r, "total_dc", m) - b.total_dc) / b.total_dc * 100);
    }
    const rhoOurs = sum.rho_ours ? sum.rho_ours["1"] : undefined;
    for (const [name, m] of Object.entries(METHODS)) {
      const r = rows.find(x => x.name === name);
      if (!r) continue;
      const save = (key, mode) => { const f = fixedAt(fixed, r.mean_r, key, mode); return f == null ? null : (f - r[key]) / f * 100; };
      const gain = key => { const f = fixedAt(fixed, r.mean_r, key, "log"); return f == null ? null : r[key] - f; };
      out.push({dataset: ds, setting: S, method: m, ks: KS[ds], cross_modal: CROSS_MODAL.has(ds),
        mean_r: r.mean_r, distinct_ef: r.distinct_ef ?? null,
        save_dc_linear: save("total_dc", "lin"), save_dc: save("total_dc", "log"),
        save_lat: r.mean_lat_us != null ? save("mean_lat_us", "log") : null,
        p1_gain: gain("p1"), p5_gain: gain("p5"),
        rho_ada: sum.rho_ada ?? null, rho_ours: rhoOurs ?? null});
    }
  }
}

const fm = (v, d = 1, s = "%") => v == null ? "n/a" : (v >= 0 ? "+" : "") + v.toFixed(d) + s;
out.sort((a, b) => a.ks - b.ks || a.dataset.localeCompare(b.dataset) || a.setting.localeCompare(b.setting));
console.log(`${"dataset".padEnd(24)} ${"KS".padStart(5)} S ${"method".padEnd(7)} ${"meanR".padStart(6)} ${"DC lin".padStart(7)} ${"DC".padStart(7)} ${"lat".padStart(7)} ${"p1 gain".padStart(7)} ${"p5 gain".padStart(7)}`);
for (const o of out)
  console.log(`${(o.dataset + (o.cross_modal ? " (x)" : "")).padEnd(24)} ${o.ks.toFixed(3)} ${o.setting} ${o.method.padEnd(7)} ` +
    `${o.mean_r.toFixed(4)} ${fm(o.save_dc_linear).padStart(7)} ${fm(o.save_dc).padStart(7)} ${fm(o.save_lat).padStart(7)} ` +
    `${fm(o.p1_gain, 3, "").padStart(7)} ${fm(o.p5_gain, 3, "").padStart(7)}`);

// Group counts. Cross-modal runs are reported separately (neither score adapts on real queries).
const groups = {
  "all, not cross-modal": o => !o.cross_modal,
  [`KS >= ${BAND_HI}, not cross-modal`]: o => !o.cross_modal && o.ks >= BAND_HI,
  [`band ${BAND_LO}-${BAND_HI}, not cross-modal`]: o => !o.cross_modal && o.ks >= BAND_LO && o.ks < BAND_HI,
  [`KS < ${BAND_LO}`]: o => !o.cross_modal && o.ks < BAND_LO,
  "cross-modal": o => o.cross_modal,
};
const cnt = (v, f) => `${v.filter(o => { const x = f(o); return x === true; }).length}/${v.filter(o => f(o) !== null).length}`;
const nz = x => x == null ? null : x;
console.log("\nCounts (dataset x setting runs; cost and p1 against a tuned fixed ef at the same mean recall, log-log)");
for (const [g, sel] of Object.entries(groups)) {
  const pick = m => out.filter(o => o.method === m && sel(o));
  const O = pick("Ours"), A = pick("Ada"), W = pick("AdaWAE");
  if (!O.length) continue;
  const pairs = O.map(o => [o, A.find(a => a.dataset === o.dataset && a.setting === o.setting)]);
  const tailBetter = pairs.filter(([o, a]) => a && o.p1_gain != null && a.p1_gain != null);
  const rhoKnown = O.filter(o => o.rho_ada != null && o.rho_ours != null);
  const worst = O.filter(o => o.save_dc != null).reduce((m, o) => Math.min(m, o.save_dc), Infinity);
  console.log(`  ${g} (${O.length} runs)`);
  console.log(`    cheaper (DC):   ours ${cnt(O, o => nz(o.save_dc) == null ? null : o.save_dc > 0)} (worst ${fm(worst)}), ` +
    `Ada ${cnt(A, o => nz(o.save_dc) == null ? null : o.save_dc > 0)}, AdaWAE ${cnt(W, o => nz(o.save_dc) == null ? null : o.save_dc > 0)}`);
  console.log(`    faster (lat):   ours ${cnt(O, o => nz(o.save_lat) == null ? null : o.save_lat > 0)}, ` +
    `Ada ${cnt(A, o => nz(o.save_lat) == null ? null : o.save_lat > 0)}, AdaWAE ${cnt(W, o => nz(o.save_lat) == null ? null : o.save_lat > 0)}`);
  console.log(`    p1 >= fixed:    ours ${cnt(O, o => nz(o.p1_gain) == null ? null : o.p1_gain >= 0)}, ` +
    `Ada ${cnt(A, o => nz(o.p1_gain) == null ? null : o.p1_gain >= 0)}, AdaWAE ${cnt(W, o => nz(o.p1_gain) == null ? null : o.p1_gain >= 0)}`);
  console.log(`    ours p1 gain >= Ada's: ${tailBetter.filter(([o, a]) => o.p1_gain >= a.p1_gain).length}/${tailBetter.length};  ` +
    `ours |rho| > Ada's: ${rhoKnown.filter(o => Math.abs(o.rho_ours) > Math.abs(o.rho_ada)).length}/${rhoKnown.length}`);
}

if (argv.includes("--loo")) {
  console.log("\nLeave-one-out test of the fixed-ef interpolation (predict each interior grid point from its two neighbours):");
  for (const m of ["lin", "log"]) {
    const v = loo[m].slice().sort((a, b) => a - b), mean = v.reduce((s, x) => s + x, 0) / v.length;
    console.log(`  ${m === "lin" ? "linear " : "log-log"}  n ${v.length}  mean ${fm(mean)}  median ${fm(v[v.length >> 1])}  range ${fm(v[0])} .. ${fm(v[v.length - 1])}`);
  }
}
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(out, null, 1));
