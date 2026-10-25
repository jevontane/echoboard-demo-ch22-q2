// Power to detect, and tell apart, three formal "morphic resonance" mechanisms in a simulated
// McDougall-style multi-generation learning experiment. Implements the plan registered on echoboard
// (study st_3f0252ccab11f51c, task ch22-q2, plan SHA-256 0b36edefeeb613d505841be792ca87cb973ec0b524654eeb85b42d5cf6cffa7b).
//
// Run: node simulate.mjs            (writes results.json; same seed, same numbers, on any machine)
import { writeFileSync } from 'node:fs';

const SEED = 20261025;
const G = 20; // generations
const TAU = 4; // saturation constant for M2/M3, in generations
const SIMILARITY = 0.5; // M3: the control line feels half the trained line's effect
const REPS = 1000;
const SIGMAS = [0.3, 0.5, 0.8];
const RS = [0.05, 0.1, 0.2];
const NS = [10, 20, 40, 80, 160, 320];
const ALPHA = 0.05;

// ---------- random numbers: mulberry32 + Box-Muller ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);
let spare = null;
function normal() {
  if (spare !== null) {
    const s = spare;
    spare = null;
    return s;
  }
  let u = 0;
  while (u === 0) u = rand();
  const v = rand();
  const r = Math.sqrt(-2 * Math.log(u));
  spare = r * Math.sin(2 * Math.PI * v);
  return r * Math.cos(2 * Math.PI * v);
}

// ---------- mechanisms: fraction of errors removed by generation g (1..G) ----------
const linear = (g) => (g - 1) / (G - 1);
const saturating = (g) => (1 - Math.exp(-(g - 1) / TAU)) / (1 - Math.exp(-(G - 1) / TAU));
const shapes = { linear, saturating };

/** One line of n rats per generation: per-generation mean and within-generation sum of squares of y = log(errors). */
function simulateLine(shape, R, sigma, n) {
  const means = new Float64Array(G);
  let withinSS = 0;
  for (let g = 1; g <= G; g++) {
    const mu = shape ? Math.log(1 - R * shape(g)) : 0;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const y = mu + sigma * normal();
      sum += y;
      sumSq += y * y;
    }
    const m = sum / n;
    means[g - 1] = m;
    withinSS += sumSq - n * m * m;
  }
  return { means, withinSS };
}

// ---------- OLS slope test on individual rats (computed from generation summaries) ----------
const gBar = (G + 1) / 2;
let sumDev2 = 0;
for (let g = 1; g <= G; g++) sumDev2 += (g - gBar) ** 2;

/** Two-sided OLS test of the slope of y on generation; returns { slope, p }. */
function slopeTest({ means, withinSS }, n) {
  const N = G * n;
  let yBar = 0;
  for (const m of means) yBar += m;
  yBar /= G;
  let sxy = 0;
  for (let g = 1; g <= G; g++) sxy += (g - gBar) * (means[g - 1] - yBar);
  const slope = sxy / sumDev2; // equals the individual-level OLS slope with equal n per generation
  const intercept = yBar - slope * gBar;
  let betweenSS = 0;
  for (let g = 1; g <= G; g++) betweenSS += n * (means[g - 1] - (intercept + slope * g)) ** 2;
  const s2 = (withinSS + betweenSS) / (N - 2);
  const t = slope / Math.sqrt(s2 / (n * sumDev2));
  return { slope, p: 2 * (1 - studentTCdf(Math.abs(t), N - 2)) };
}

// ---------- Student t CDF via the regularised incomplete beta function (Numerical Recipes) ----------
function logGamma(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x;
  const tmp = x + 5.5 - (x + 0.5) * Math.log(x + 5.5);
  let ser = 1.000000000190015;
  for (const ci of c) ser += ci / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
function betacf(a, b, x) {
  const EPS = 3e-16;
  const FPMIN = 1e-300;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}
function incompleteBeta(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}
function studentTCdf(t, df) {
  const x = df / (df + t * t);
  const tail = 0.5 * incompleteBeta(df / 2, 0.5, x);
  return t >= 0 ? 1 - tail : tail;
}

// ---------- discrimination: fit linear vs saturating to generation means (R free, intercept free) ----------
function sseFor(means, shape, R) {
  // For a given R the best intercept is the mean residual; return the residual sum of squares.
  let c = 0;
  const pred = new Float64Array(G);
  for (let g = 1; g <= G; g++) {
    pred[g - 1] = Math.log(1 - R * shape(g));
    c += means[g - 1] - pred[g - 1];
  }
  c /= G;
  let sse = 0;
  for (let g = 1; g <= G; g++) sse += (means[g - 1] - c - pred[g - 1]) ** 2;
  return sse;
}
function bestSSE(means, shape) {
  // Golden-section search for R in [-0.95, 0.95] (allows apparent worsening as well as improvement).
  let lo = -0.95;
  let hi = 0.95;
  const phi = (Math.sqrt(5) - 1) / 2;
  let x1 = hi - phi * (hi - lo);
  let x2 = lo + phi * (hi - lo);
  let f1 = sseFor(means, shape, x1);
  let f2 = sseFor(means, shape, x2);
  for (let i = 0; i < 60; i++) {
    if (f1 < f2) {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = hi - phi * (hi - lo);
      f1 = sseFor(means, shape, x1);
    } else {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = lo + phi * (hi - lo);
      f2 = sseFor(means, shape, x2);
    }
  }
  return Math.min(f1, f2);
}
const pickLinear = (means) => bestSSE(means, linear) < bestSSE(means, saturating);

// ---------- n80: interpolate power against log2(n) ----------
function n80(powers) {
  for (let i = 0; i < NS.length; i++) {
    if (powers[i] >= 0.8) {
      if (i === 0) return NS[0];
      const [x0, x1] = [Math.log2(NS[i - 1]), Math.log2(NS[i])];
      const [y0, y1] = [powers[i - 1], powers[i]];
      return Math.round(2 ** (x0 + ((0.8 - y0) * (x1 - x0)) / (y1 - y0)));
    }
  }
  return null; // above the largest n tested
}

// ---------- run the grid ----------
const detected = (line, n) => {
  const { slope, p } = slopeTest(line, n);
  return p < ALPHA && slope < 0;
};
const cells = [];
const started = Date.now();
for (const sigma of SIGMAS) {
  for (const R of RS) {
    for (const n of NS) {
      let falsePos = 0;
      let m1 = 0;
      let m2 = 0;
      let m3control = 0;
      let disc = 0;
      for (let rep = 0; rep < REPS; rep++) {
        const base = simulateLine(null, 0, sigma, n);
        if (slopeTest(base, n).p < ALPHA) falsePos++;
        const l1 = simulateLine(linear, R, sigma, n);
        if (detected(l1, n)) m1++;
        const l2 = simulateLine(saturating, R, sigma, n);
        if (detected(l2, n)) m2++;
        // M3: trained line behaves as M2; the control line feels SIMILARITY of its effect.
        const control = simulateLine(saturating, SIMILARITY * R, sigma, n);
        if (detected(control, n)) m3control++;
        if (pickLinear(l1.means)) disc++;
        if (!pickLinear(l2.means)) disc++;
      }
      cells.push({
        sigma,
        R,
        n,
        falsePositiveRate: falsePos / REPS,
        powerM1: m1 / REPS,
        powerM2: m2 / REPS,
        powerM3Control: m3control / REPS,
        discriminationM1vsM2: disc / (2 * REPS),
      });
    }
  }
  process.stderr.write(`sigma ${sigma} done (${Math.round((Date.now() - started) / 1000)} s)\n`);
}

// ---------- n80 per (sigma, R) and the registered hypotheses ----------
const summaries = [];
for (const sigma of SIGMAS) {
  for (const R of RS) {
    const row = cells.filter((c) => c.sigma === sigma && c.R === R);
    summaries.push({
      sigma,
      R,
      n80DetectM1: n80(row.map((c) => c.powerM1)),
      n80DetectM2: n80(row.map((c) => c.powerM2)),
      n80DetectM3Control: n80(row.map((c) => c.powerM3Control)),
      n80DiscriminateM1vsM2: n80(row.map((c) => c.discriminationM1vsM2)),
    });
  }
}
const key = summaries.find((s) => s.sigma === 0.5 && s.R === 0.1);
const verdict = (v) => (v === null ? 'inconclusive' : v ? 'supported' : 'disconfirmed');
const H1 = key.n80DetectM1 === null ? null : key.n80DetectM1 >= 75 && key.n80DetectM1 <= 125;
const H2 = key.n80DiscriminateM1vsM2 === null ? null : key.n80DetectM1 === null ? null : key.n80DiscriminateM1vsM2 > key.n80DetectM1;
const ratio = key.n80DetectM3Control !== null && key.n80DetectM2 !== null ? key.n80DetectM3Control / key.n80DetectM2 : null;
const H3 = ratio === null ? null : ratio >= 3 && ratio <= 5;
const verdicts = [H1, H2, H3];
const outcome = verdicts.every((v) => v === true) ? 'positive' : verdicts.every((v) => v === false) ? 'null' : 'inconclusive';

const results = {
  study: 'st_3f0252ccab11f51c',
  task: 'ch22-q2',
  planSha256: '0b36edefeeb613d505841be792ca87cb973ec0b524654eeb85b42d5cf6cffa7b',
  settings: { seed: SEED, generations: G, tau: TAU, similarity: SIMILARITY, reps: REPS, sigmas: SIGMAS, Rs: RS, ns: NS, alpha: ALPHA },
  hypotheses: {
    H1: { claim: 'M1 detection n80 at R=0.10, sigma=0.5 is in 75-125', n80: key.n80DetectM1, verdict: verdict(H1) },
    H2: { claim: 'M1-vs-M2 discrimination n80 > M1 detection n80', n80Discrimination: key.n80DiscriminateM1vsM2, n80Detection: key.n80DetectM1, verdict: verdict(H2) },
    H3: { claim: 'M3 control-line n80 / M2 trained-line n80 is in 3-5', ratio: ratio === null ? null : Number(ratio.toFixed(2)), n80Control: key.n80DetectM3Control, n80Trained: key.n80DetectM2, verdict: verdict(H3) },
  },
  outcome,
  n80: summaries,
  cells,
  seconds: Math.round((Date.now() - started) / 1000),
};
writeFileSync('results.json', JSON.stringify(results, null, 2) + '\n');
console.log(JSON.stringify({ hypotheses: results.hypotheses, outcome, n80Key: key }, null, 2));
