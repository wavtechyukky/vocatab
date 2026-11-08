// フレームごとのパラメータの定義と、曲線の計算・編集。
import { t } from "./i18n";
import type { Curve, Interp, ParamKey } from "./model";

export interface ParamDef {
  key: ParamKey;
  label: string;
  min: number;
  max: number;
  def: number;
  viewMin: number; // パネルに出す範囲
  viewMax: number;
  top: string; // 縦軸の上端と下端の言葉
  bottom: string;
  voiceParam: boolean; // NSF-HiFiGAN では効かない
  fmt: (v: number) => string;
}

// 名前と縦軸の言葉は画面の言語で（i18n.ts は読み込んだ時点で言語を決めているので、ここで訳してよい）
export const PARAMS: ParamDef[] = [
  { key: "pitchDelta", label: t("ピッチのずれ"), min: -1200, max: 1200, def: 0, viewMin: -300, viewMax: 300,
    top: "+300 ¢", bottom: "−300 ¢", voiceParam: false, fmt: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(0)} ¢` },
  { key: "intonationEnv", label: t("抑揚の強さ"), min: 0, max: 2, def: 1, viewMin: 0, viewMax: 2,
    top: "× 2", bottom: "× 0", voiceParam: false, fmt: (v) => `× ${v.toFixed(2)}` },
  { key: "vibratoEnv", label: t("ビブラートの強さ"), min: 0, max: 2, def: 1, viewMin: 0, viewMax: 2,
    top: "× 2", bottom: "× 0", voiceParam: false, fmt: (v) => `× ${v.toFixed(2)}` },
  { key: "loudness", label: t("音量"), min: -48, max: 12, def: 0, viewMin: -24, viewMax: 12,
    top: "+12 dB", bottom: "−24 dB", voiceParam: false, fmt: (v) => `${v.toFixed(1)} dB` },
  { key: "tension", label: t("テンション"), min: -1, max: 1, def: 0, viewMin: -1, viewMax: 1,
    top: t("張る"), bottom: t("緩む"), voiceParam: true, fmt: (v) => v.toFixed(2) },
  { key: "breathiness", label: t("ブレシネス"), min: -1, max: 1, def: 0, viewMin: -1, viewMax: 1,
    top: t("息が多い"), bottom: t("息が少ない"), voiceParam: true, fmt: (v) => v.toFixed(2) },
  { key: "gender", label: t("ジェンダー"), min: -1, max: 1, def: 0, viewMin: -1, viewMax: 1,
    top: t("男性的"), bottom: t("女性的"), voiceParam: true, fmt: (v) => v.toFixed(2) },
  { key: "voicing", label: t("有声度"), min: 0, max: 1, def: 1, viewMin: 0, viewMax: 1,
    top: t("有声"), bottom: t("無声"), voiceParam: true, fmt: (v) => v.toFixed(2) },
  { key: "consonant", label: t("子音の強さ"), min: -1, max: 1, def: 0, viewMin: -1, viewMax: 1,
    top: t("強い"), bottom: t("弱い"), voiceParam: true, fmt: (v) => v.toFixed(2) },
  { key: "resonance", label: t("響き"), min: -1, max: 1, def: 0, viewMin: -1, viewMax: 1,
    top: t("響く"), bottom: t("こもる"), voiceParam: true, fmt: (v) => v.toFixed(2) },
  { key: "growl", label: t("うなり"), min: 0, max: 1, def: 0, viewMin: 0, viewMax: 1,
    top: t("強い"), bottom: t("なし"), voiceParam: false, fmt: (v) => v.toFixed(2) },
  // 滑舌: 子音の凹み（音素パネルの谷）の既定の深さに掛ける倍率。0 = 凹み無し（既定）、1 = ボイスバンクのレシピどおり、
  // 2 = その 2 倍。手で動かした凹みには掛けない。音響モデルの前で効く。トラック全体の値と曲線は足し合わせる
  { key: "articulation", label: t("滑舌"), min: 0, max: 2, def: 0, viewMin: 0, viewMax: 2,
    top: "× 2", bottom: "× 0", voiceParam: false, fmt: (v) => `× ${v.toFixed(2)}` },
];

/** エンジンへ渡すボイスパラメータ（ピッチのずれとビブラートの強さは F0 の組み立てで使うので除く）。 */
export const VOICE_KEYS: ParamKey[] = ["tension", "breathiness", "gender", "voicing", "loudness",
                                       "consonant", "resonance", "growl", "articulation"];

/** トラック全体の値と曲線を掛け合わせるパラメータ（ほかは足し合わせる）。既定値は 1 */
export const MULTIPLY_KEYS: ReadonlySet<ParamKey> = new Set<ParamKey>(["voicing"]);

export const paramDef = (key: ParamKey): ParamDef => PARAMS.find((p) => p.key === key)!;

function shape(interp: Interp, y0: number, y1: number, u: number, ym1: number, y2: number): number {
  if (interp === "linear") return y0 + (y1 - y0) * u;
  if (interp === "cosine") return y0 + ((y1 - y0) * (1 - Math.cos(Math.PI * u))) / 2;
  // Catmull-Rom（端は隣の点を繰り返す）
  const u2 = u * u;
  const u3 = u2 * u;
  return 0.5 * (2 * y0 + (-ym1 + y1) * u + (2 * ym1 - 5 * y0 + 4 * y1 - y2) * u2 + (-ym1 + 3 * y0 - 3 * y1 + y2) * u3);
}

/** 刻み t での値。点が無い所（最初の点より前・最後の点より後）は既定値。 */
export function evalCurve(c: Curve | undefined, t: number, def: number): number {
  if (!c || c.pts.length === 0) return def;
  const p = c.pts;
  if (t < p[0][0] || t > p[p.length - 1][0]) return def;
  if (p.length === 1) return p[0][1];
  let lo = 0;
  let hi = p.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (p[mid][0] <= t) lo = mid;
    else hi = mid;
  }
  const [x0, y0] = p[lo];
  const [x1, y1] = p[hi];
  const u = x1 > x0 ? (t - x0) / (x1 - x0) : 0;
  const ym1 = lo > 0 ? p[lo - 1][1] : y0;
  const y2 = hi + 1 < p.length ? p[hi + 1][1] : y1;
  return shape(c.interp, y0, y1, u, ym1, y2);
}

export function ensureCurve(params: Partial<Record<ParamKey, Curve>>, key: ParamKey): Curve {
  let c = params[key];
  if (!c) {
    c = { pts: [], interp: "cubic" };
    params[key] = c;
  }
  return c;
}

/** [a, b] の中の点を消す（両端を含む）。 */
export function removeRange(c: Curve, a: number, b: number): void {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  c.pts = c.pts.filter(([t]) => t < lo || t > hi);
}

export function insertPoint(c: Curve, t: number, v: number): void {
  const i = c.pts.findIndex(([x]) => x >= t);
  if (i < 0) c.pts.push([t, v]);
  else if (c.pts[i][0] === t) c.pts[i] = [t, v];
  else c.pts.splice(i, 0, [t, v]);
}

/** 消しゴム: [a, b] を既定値に戻す（外側に点が残っていれば、境目に既定値の点を置いて段差を作らない）。 */
export function eraseRange(c: Curve, a: number, b: number, def: number): void {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  removeRange(c, lo, hi);
  if (c.pts.some(([t]) => t < lo)) insertPoint(c, lo, def);
  if (c.pts.some(([t]) => t > hi)) insertPoint(c, hi, def);
}

/**
 * 鉛筆の一筆ぶん: 前の位置 (t0, v0) から今の位置 (t1, v1) までの点を置き換える。
 * minGap より近い点は置かない（点が増えすぎないように）。
 */
export function penSegment(c: Curve, t0: number, v0: number, t1: number, v1: number, minGap: number): void {
  removeRange(c, t0, t1);
  insertPoint(c, t0, v0);
  const n = Math.floor(Math.abs(t1 - t0) / minGap);
  for (let k = 1; k <= n; k++) {
    const u = k / (n + 1);
    insertPoint(c, t0 + (t1 - t0) * u, v0 + (v1 - v0) * u);
  }
  insertPoint(c, t1, v1);
}

export const clampParam = (d: ParamDef, v: number): number => Math.min(d.max, Math.max(d.min, v));

// ── 一筆で範囲だけを書き換える ─────────────────────────
// 曲線は点と点の間を補間するので、描いた所の点を置き換えるだけだと、一筆の端と遠くの別の点が補間でつながり、
// 描いていない所（隣以降の音符）まで値が変わってしまう。そこで一筆が変えるのは [lo, hi] だけにし、
// 外側は元の曲線の形のまま残す（境目のすぐ外に元の値を写した点を置く）。
const EDGE = 1; // 境目に置く点の間隔（刻み）
type Pt = [number, number];

/** 元の曲線の (a, b] を細かく取った点（補間の形をほぼそのまま写す）。 */
function sampleOrig(orig: Curve, a: number, b: number, def: number, out: Pt[]): void {
  const n = Math.min(64, Math.max(1, Math.ceil((b - a) / 20)));
  for (let k = 1; k <= n; k++) {
    const t = a + ((b - a) * k) / n;
    out.push([t, evalCurve(orig, t, def)]);
  }
}

/**
 * 曲線の [lo, hi] を pts で置き換え、外側は元（base）のまま残す。
 * 元の点の間にまたがる区間は元の形を細かく写し、元の点の外（既定値の所）は既定値の点を 2 つずつ置いて平らに保つ。
 */
export function replaceRange(c: Curve, base: Pt[], lo: number, hi: number, pts: Pt[], def: number): void {
  const orig: Curve = { pts: base, interp: c.interp };
  const out: Pt[] = base.filter(([t]) => t < lo - 2 * EDGE || t > hi + 2 * EDGE);
  const L = [...base].reverse().find(([t]) => t < lo - 2 * EDGE);
  const R = base.find(([t]) => t > hi + 2 * EDGE);
  // 境目のすぐ外には元の値の点を 2 つ並べる（描いた値との段差による補間の行き過ぎを、その 1 刻みの間に閉じ込める）
  if (L) {
    if (base.some(([t]) => t >= lo - 2 * EDGE)) {
      sampleOrig(orig, L[0], lo - 2 * EDGE, def, out);
      out.push([lo - EDGE, evalCurve(orig, lo - EDGE, def)]);
    } else out.push([L[0] + EDGE, def], [L[0] + 2 * EDGE, def], [lo - 2 * EDGE, def], [lo - EDGE, def]);
  }
  if (R) {
    if (base.some(([t]) => t <= hi + 2 * EDGE)) {
      const tmp: Pt[] = [];
      sampleOrig(orig, hi + 2 * EDGE, R[0], def, tmp);
      out.push([hi + EDGE, evalCurve(orig, hi + EDGE, def)], [hi + 2 * EDGE, evalCurve(orig, hi + 2 * EDGE, def)],
               ...tmp.slice(0, -1));
    } else {
      out.push([hi + EDGE, def], [hi + 2 * EDGE, def], [R[0] - 2 * EDGE, def], [R[0] - EDGE, def]);
    }
  }
  for (const p of pts) if (p[0] >= lo && p[0] <= hi) out.push(p);
  out.sort((a, b) => a[0] - b[0]);
  // 同じ刻み・逆順の点を落とす（境目の点どうしが重なったとき）
  c.pts = out.filter((p, i) => i === 0 || p[0] > out[i - 1][0]);
}

/** 鉛筆の一筆。add のたびに、描いた範囲だけを描いた値で置き換える（外側は描き始めの前の形のまま）。 */
export class PenStroke {
  private readonly base: Pt[];
  private readonly buckets = new Map<number, Pt>(); // gap ごとに 1 点（同じ所を描き直したら新しい値）
  private last: Pt | null = null;
  private lo = Infinity;
  private hi = -Infinity;

  constructor(private c: Curve, private def: number, private gap: number) {
    this.base = c.pts.map(([t, v]) => [t, v] as Pt);
  }

  /** connected = false: 前の点と線でつながない（声の無い所を飛ばした後など） */
  add(t: number, v: number, connected = true): void {
    const put = (tt: number, vv: number) => this.buckets.set(Math.round(tt / this.gap), [tt, vv]);
    if (this.last && connected) {
      const [t0, v0] = this.last;
      const n = Math.floor(Math.abs(t - t0) / this.gap);
      for (let k = 1; k <= n; k++) {
        const u = k / (n + 1);
        put(t0 + (t - t0) * u, v0 + (v - v0) * u);
      }
    }
    put(t, v);
    this.last = [t, v];
    this.lo = Math.min(this.lo, t);
    this.hi = Math.max(this.hi, t);
    const pts = [...this.buckets.values()].sort((a, b) => a[0] - b[0]);
    replaceRange(this.c, this.base, pts[0][0], pts[pts.length - 1][0], pts, this.def);
  }

  /**
   * 一筆を終える。描いた幅が minWidth（刻み）より狭ければ、最後の値を minWidth の幅で保つ。
   * 縦にまっすぐ描いた所やごく短い一筆が、エンジンのフレーム（約 5.8 ms）より細いトゲになり、
   * 聞こえなくなるのを防ぐ（エンジンはフレームの頭の値を読み、さらに数フレームでならす）。
   */
  finish(minWidth: number): void {
    if (!this.last || this.hi - this.lo >= minWidth) return;
    const mid = (this.lo + this.hi) / 2;
    const half = minWidth / 2;
    const v = this.last[1];
    replaceRange(this.c, this.base, mid - half, mid + half, [[mid - half, v], [mid + half, v]], this.def);
  }
}

/** 消しゴムの一筆: なぞった範囲を既定値に戻す（外側は消し始めの前の形のまま）。 */
export class EraseStroke {
  private readonly base: Pt[];
  private lo = Infinity;
  private hi = -Infinity;

  constructor(private c: Curve, private def: number) {
    this.base = c.pts.map(([t, v]) => [t, v] as Pt);
  }

  add(t: number): void {
    this.lo = Math.min(this.lo, t);
    this.hi = Math.max(this.hi, t);
    const d = this.def;
    // 中は既定値で平ら（端に 2 点ずつ置いて、補間が膨らまないように）
    const inner: Pt[] = this.hi - this.lo > 4 * EDGE
      ? [[this.lo, d], [this.lo + EDGE, d], [this.hi - EDGE, d], [this.hi, d]] : [[this.lo, d], [this.hi, d]];
    replaceRange(this.c, this.base, this.lo, this.hi, inner, d);
    // 点が既定値だけになったら空にする（曲線が無いのと同じ）
    if (this.c.pts.every(([, v]) => v === d)) this.c.pts = [];
  }
}

/**
 * 直線: (t0, v0) から (t1, v1) までを直線で置き換える（補間で曲がらないよう細かく点を置く）。外側は元のまま。
 * 幅が minWidth（刻み）より狭い直線（縦に引いたもの）は、v1 を minWidth の幅で保つ。
 */
export function lineRange(c: Curve, t0: number, v0: number, t1: number, v1: number, gap: number, def: number,
                          minWidth = 0): void {
  const s = new PenStroke(c, def, gap);
  if (Math.abs(t1 - t0) < minWidth) {
    s.add(t1, v1, false);
    s.finish(minWidth);
    return;
  }
  s.add(t0, v0, false);
  s.add(t1, v1, true);
}
