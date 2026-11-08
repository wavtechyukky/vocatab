// 子音の凹み（擬似 mel の対数レベル整形）: engine/x0shape.py の移植。音響モデルの x0_loggain 入力
// （フレームごとの ln(gain)、0 = 素のまま）に渡す曲線を、音素列と持続長から作る。
// 数値は Python と同じ順に double で計算し、Float32Array に入れる所だけ float32 に丸める（Python 側も同じ形にしてある）。
// ui/test/x0shape.test.ts と synth.test.ts で Python の値と照らし合わせる。

interface Classes {
  stop: Set<string>;
  affr: Set<string>;
  fric_nu: Set<string>;
  sonorant: Set<string>;
  voiced_stop: Set<string>;
  nasal: Set<string>;
  cy: Set<string>;
}

/** 言語ごとの音素の分類（engine/x0shape.py の CLASSES） */
const CLASSES: Record<string, Classes> = {
  ja: {
    stop: new Set(["k", "ky", "t", "ty", "p", "py", "b", "by", "d", "dy", "g", "gy", "cl"]),
    affr: new Set(["ch", "ts"]),
    fric_nu: new Set(["z", "j"]),
    sonorant: new Set(["n", "ny", "m", "my", "y", "r", "ry", "w", "v"]),
    voiced_stop: new Set(["b", "by", "d", "dy", "g", "gy"]),
    nasal: new Set(["N", "n", "ny", "m", "my"]),
    cy: new Set(["ky", "gy", "ty", "dy", "py", "by", "fy", "hy"]),
  },
  en: {
    stop: new Set(["b", "d", "dx", "g", "k", "p", "q", "t", "cl"]),
    affr: new Set(["ch", "tr"]),
    fric_nu: new Set(["z", "zh", "v", "dh", "jh", "dr"]),
    sonorant: new Set(["m", "n", "ng", "l", "r", "w", "y"]),
    voiced_stop: new Set(["b", "d", "g", "dx"]),
    nasal: new Set(["m", "n", "ng"]),
    cy: new Set(),
  },
};

/** 既定値（engine/x0shape.py の DEFAULTS）。voicebank.yaml の x0_shape がこれを上書きする */
export const DEFAULTS: Record<string, any> = {
  burst: true, burst_shape: "nu", W: 6, floor: 0.891, pos_stop: 0.30, pos_affr: 0.10, closure_max: 0,
  cy_frac: 0.55, nasal_exc: true, fric: true, fric_depth: 0.891, son: true, son_depth: 0.891,
};
export const MIN_DIP = 3; // ν 字の谷の最短のフレーム数
export const DIP_DB_MIN = -5.0; // 画面で動かせる凹みの深さの限界（名目 dB。engine/pipeline.py の DIP_DB_MIN）

export type Dip = { amp: number; w: number };
type Seg = [string, number, number]; // 音素・頭のフレーム・終わりのフレーム
type BurstPlan = Map<number, [string, number, number, number]>; // 形・谷の終わり r・振幅・ランプの幅

/** Python の round（ちょうど半分は偶数へ） */
function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export function classesOf(language: string): Classes | null {
  return CLASSES[language] ?? CLASSES[String(language).split("_")[0]] ?? null;
}

/** voicebank.yaml の x0_shape → 既定値で埋めた設定。整形しない（enabled: false）なら null */
export function paramsOf(cfg: Record<string, any> | null | undefined): Record<string, any> | null {
  const p = { ...DEFAULTS, ...(cfg ?? {}) };
  return (p.enabled ?? true) ? p : null;
}

function pp(p: Record<string, any>, ph: string): Record<string, any> {
  const o = (p.per_ph ?? {})[ph];
  return o ? { ...p, ...o } : p;
}

/** g[s:e] を v0 → v1 の余弦ランプにする（np.linspace(0, 1, n + 1)[1:] と同じ点） */
function ramp(g: Float32Array, s: number, e: number, v0: number, v1: number): void {
  const n = e - s;
  if (n <= 0) return;
  const step = 1 / n;
  for (let k = 1; k <= n; k++) {
    const x = k === n ? 1 : k * step;
    g[s + k - 1] = v0 + ((v1 - v0) * (1 - Math.cos(Math.PI * x))) / 2;
  }
}

/** g[s:e] に Hann 窓型（ν 字）の谷を掛ける。谷の底の振幅 = depth */
function hannDip(g: Float32Array, s: number, e: number, depth: number): void {
  const n = e - s;
  if (n < MIN_DIP) return;
  for (let k = 0; k < n; k++) {
    const x = (k + 0.5) / n;
    g[s + k] = g[s + k] * (1 - ((1 - depth) * (1 - Math.cos(2 * Math.PI * x))) / 2);
  }
}

function burstPlan(segs: Seg[], c: Classes, p: Record<string, any>): BurstPlan {
  const plan: BurstPlan = new Map();
  const W = Math.trunc(Number(p.W));
  const cyFrac = Number(p.cy_frac);
  let prev: string | null = null;
  segs.forEach(([ph, s, e], i) => {
    const cls = c.stop.has(ph) ? "stop" : c.affr.has(ph) ? "affr" : null;
    if (cls === null) {
      prev = ph;
      return;
    }
    if ((p.nasal_exc ?? true) && c.voiced_stop.has(ph) && prev !== null && c.nasal.has(prev)) {
      prev = ph; // 例: ん+が。閉鎖は有声・鼻腔経由なので触らない
      return;
    }
    const q = pp(p, ph);
    prev = ph;
    if (q.off) return;
    const floor = Number(q.floor);
    const eEff = c.cy.has(ph) ? s + Math.max(2, roundHalfEven(cyFrac * (e - s))) : e;
    const shape = q.shape || (ph !== "cl" ? q.burst_shape : undefined);
    if (ph === "cl") { // 閉鎖のみ。開放は次の子音の整形が受け持つ
      if (shape === "nu") plan.set(i, ["nu", e, Number(q.depth ?? floor), 0]);
      else plan.set(i, ["cl_trap", e, floor, Math.max(1, Math.min(W, Math.floor((e - s) / 3)))]);
      return;
    }
    const pv = Number(cls === "stop" ? q.pos_stop : q.pos_affr);
    const cap = Math.trunc(Number(q.closure_max ?? 0) || 0);
    if (shape === "nu") { // ν 字の谷。開放位置 r までで復帰（床なし）
      let r = Math.min(s + Math.max(3, roundHalfEven(pv * (eEff - s))), eEff);
      if (cap > 0) r = Math.min(r, s + Math.max(3, cap));
      plan.set(i, ["nu", r, Number(q.depth ?? floor), 0]);
      return;
    }
    const w = Math.max(1, Math.min(W, Math.floor((eEff - s) / 3))); // 台形はランプの分 w が開放位置の下限になる
    let r = Math.min(s + Math.max(w, roundHalfEven(pv * (eEff - s))), eEff);
    if (cap > 0) r = Math.min(r, s + Math.max(w, cap));
    plan.set(i, ["trap", r, floor, w]);
  });
  return plan;
}

function burst(T: number, segs: Seg[], plan: BurstPlan, skip: Set<number>): Float32Array {
  const g = new Float32Array(T).fill(1);
  segs.forEach(([, s, e], i) => {
    const pl = plan.get(i);
    if (!pl || skip.has(i)) return;
    const [kind, r, amp, w] = pl;
    const v0 = s > 0 ? g[s - 1] : 1;
    if (kind === "nu") hannDip(g, s, r, amp);
    else if (kind === "cl_trap") {
      ramp(g, s, s + w, v0, amp);
      for (let t = s + w; t < e; t++) g[t] = amp;
    } else {
      ramp(g, s, s + w, v0, amp);
      for (let t = s + w; t < Math.max(s + w, r - w); t++) g[t] = amp;
      ramp(g, Math.max(s + w, r - w), r, amp, 1);
    }
  });
  return g;
}

function nuPlan(segs: Seg[], set: Set<string>, key: string, p: Record<string, any>): Map<number, number> {
  const plan = new Map<number, number>();
  segs.forEach(([ph], i) => {
    if (!set.has(ph)) return;
    const q = pp(p, ph);
    if (q.off) return;
    plan.set(i, Number(q.depth ?? q[key]));
  });
  return plan;
}

function segsOf(phonemes: string[], phDurs: ArrayLike<number>): { segs: Seg[]; T: number } {
  const segs: Seg[] = [];
  let pos = 0;
  phonemes.forEach((ph, j) => {
    const d = Number(phDurs[j]);
    segs.push([ph, pos, pos + d]);
    pos += d;
  });
  return { segs, T: pos };
}

function plans(segs: Seg[], c: Classes, p: Record<string, any>) {
  return {
    burst: (p.burst ?? true) ? burstPlan(segs, c, p) : null,
    fric: (p.fric ?? true) ? nuPlan(segs, c.fric_nu, "fric_depth", p) : null,
    son: (p.son ?? true) ? nuPlan(segs, c.sonorant, "son_depth", p) : null,
  };
}

/**
 * 子音ごとの既定の谷: 音素ごとに {amp: 谷の底の振幅（1 = 凹みなし）, w: 広さ（音素の長さに対する割合）}、子音でなければ null。
 * isConsonant(音素) は言語パックの判定（pau・核でない音素）
 */
export function dips(phonemes: string[], phDurs: ArrayLike<number>, cfg: Record<string, any> | null | undefined,
                     language: string, isConsonant: (ph: string) => boolean): Array<Dip | null> {
  const c = classesOf(language);
  const p = paramsOf(cfg);
  const { segs } = segsOf(phonemes, phDurs);
  const out: Array<Dip | null> = segs.map(([ph]) => (isConsonant(ph) ? { amp: 1, w: 1 } : null));
  if (!c || !p) return out;
  const pl = plans(segs, c, p);
  segs.forEach(([, s, e], i) => {
    if (out[i] === null || e <= s) return;
    const b = pl.burst?.get(i);
    if (b) out[i] = { amp: b[2], w: (b[1] - s) / (e - s) };
    else if (pl.fric?.has(i)) out[i] = { amp: pl.fric.get(i)!, w: 1 };
    else if (pl.son?.has(i)) out[i] = { amp: pl.son.get(i)!, w: 1 };
  });
  return out;
}

/**
 * 画面で動かした凹み {句の中の音符の番号: {音符の中の音素の番号: {db, w}}} → loggain の overrides {句の中の音素の番号: {amp, w}}。
 * 片方だけのときは既定で埋める（engine/pipeline.py の dip_overrides）
 */
export function dipOverrides(dipsArr: Array<Dip | null> | null, noteIndex: number[],
                             phDips: Record<number, Record<number, { db?: number; w?: number }>>): Map<number, Dip> {
  const out = new Map<number, Dip>();
  if (!dipsArr || !Object.keys(phDips).length) return out;
  const count = new Map<number, number>();
  noteIndex.forEach((ni, j) => {
    if (ni < 0) return;
    const idx = count.get(ni) ?? 0;
    count.set(ni, idx + 1);
    const o = phDips[ni]?.[idx];
    const d = dipsArr[j];
    if (!o || !d) return;
    const amp = o.db === undefined || o.db === null ? d.amp : 10 ** (Math.min(0, Math.max(DIP_DB_MIN, Number(o.db))) / 20);
    const w = o.w === undefined || o.w === null ? d.w : Math.min(1, Math.max(0, Number(o.w)));
    out.set(j, { amp, w });
  });
  return out;
}

/**
 * 滑舌（フレームごとの倍率）→ 子音ごとの既定の凹みの倍率 {句の中の音素の番号: 倍率}。子音の真ん中のフレームの値を使い、
 * −5 dB より深くしない。手で動かした子音と倍率 1 の子音は入れない（engine/pipeline.py の dip_scales）
 */
export function dipScales(dipsArr: Array<Dip | null> | null, phDurs: ArrayLike<number>, articulation: ArrayLike<number>,
                          ov: Map<number, Dip>): Map<number, number> {
  const out = new Map<number, number>();
  if (!dipsArr) return out;
  let f = 0;
  for (let j = 0; j < phDurs.length; j++) {
    const d = Number(phDurs[j]);
    const dip = dipsArr[j];
    if (dip && !ov.has(j) && d > 0 && articulation.length) {
      let m = Math.min(2, Math.max(0, articulation[Math.min(articulation.length - 1, f + Math.floor(d / 2))]));
      const db0 = 20 * Math.log10(Math.max(dip.amp, 1e-4));
      if (db0 < 0) m = Math.min(m, DIP_DB_MIN / db0);
      if (m !== 1) out.set(j, m);
    }
    f += d;
  }
  return out;
}

/**
 * 音素列 + 持続長（フレーム）→ ln(gain) [T]。整形しないなら null。
 * overrides = {句の中の音素の番号: {amp, w}}（画面で動かした凹み。その音素は頭から w の区間の ν 字の谷になる）、
 * scales = {句の中の音素の番号: 倍率}（既定の谷の深さを dB で何倍にするか。上書きした音素には掛けない）
 */
export function loggain(phonemes: string[], phDurs: ArrayLike<number>, cfg: Record<string, any> | null | undefined,
                        language: string, overrides?: Map<number, Dip> | null,
                        scales?: Map<number, number> | null): Float32Array | null {
  const c = classesOf(language);
  const p = paramsOf(cfg);
  if (!c || !p) return null;
  const { segs, T } = segsOf(phonemes, phDurs);
  const ov = new Map([...(overrides ?? new Map<number, Dip>())].filter(([k]) => k >= 0 && k < segs.length));
  const pl = plans(segs, c, p);
  if (scales && scales.size) {
    // 振幅 amp → amp ** m（dB で m 倍）。m = 1 の音素は触らない
    const sc = (i: number) => (scales.has(i) && scales.get(i) !== 1 ? scales.get(i)! : null);
    if (pl.burst) for (const [i, v] of pl.burst) { const m = sc(i); if (m !== null) pl.burst.set(i, [v[0], v[1], v[2] ** m, v[3]]); }
    for (const plan of [pl.fric, pl.son]) if (plan) for (const [i, a] of plan) { const m = sc(i); if (m !== null) plan.set(i, a ** m); }
  }
  const curves: Float32Array[] = [];
  if (pl.burst) curves.push(burst(T, segs, pl.burst, new Set(ov.keys())));
  for (const plan of [pl.fric, pl.son]) {
    if (!plan) continue;
    const g = new Float32Array(T).fill(1);
    for (const [i, amp] of plan) if (!ov.has(i)) hannDip(g, segs[i][1], segs[i][2], amp);
    curves.push(g);
  }
  if (ov.size) {
    const g = new Float32Array(T).fill(1);
    for (const i of [...ov.keys()].sort((a, b) => a - b)) {
      const o = ov.get(i)!;
      const [, s, e] = segs[i];
      const n = Math.min(e - s, Math.max(MIN_DIP, roundHalfEven(o.w * (e - s))));
      hannDip(g, s, s + n, o.amp);
    }
    curves.push(g);
  }
  if (!curves.length) return null;
  let g = curves[0];
  for (const x of curves.slice(1)) {
    const y = new Float32Array(T);
    for (let t = 0; t < T; t++) y[t] = g[t] * x[t];
    g = y;
  }
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) out[t] = Math.log(Math.max(g[t], 1e-4));
  return out;
}
