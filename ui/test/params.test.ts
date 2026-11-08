// ボイスパラメータの変化が合成に届くか（「急に変えても効かないことがある」の再発防止）。node で動く。
//   npx esbuild@0.25.0 ui/test/params.test.ts --bundle --platform=node --format=esm --outfile=<作業用>/t.mjs && node <作業用>/t.mjs
// 1. 合成し直すかどうかの指紋（hashF32）が、符号だけ違う配列・0 と 1・x と 2x で衝突しない
// 2. 縦にまっすぐ描いた一筆・縦に引いた直線が、エンジンのフレーム（頭の値を読む）で拾える幅になる
// 3. 速く動かした一筆の山が、点の間隔で取りこぼされない
import { hashF32 } from "../src/engine";
import { evalCurve, lineRange, PenStroke } from "../src/params";
import { type Curve, FPS, secPerTick } from "../src/model";
import { done, same } from "./golden";

// ── 1. 指紋 ─────────────────────────────────
const filled = (n: number, v: number) => new Float32Array(n).fill(v);
same("符号だけ違う（偶数の長さ）", hashF32(filled(100, 0.5), "k") !== hashF32(filled(100, -0.5), "k"), true);
same("+1 と −1", hashF32(filled(64, 1), "k") !== hashF32(filled(64, -1), "k"), true);
same("0 と 1.0", hashF32(filled(64, 0), "k") !== hashF32(filled(64, 1), "k"), true);
same("x と 2x", hashF32(filled(64, 0.25), "k") !== hashF32(filled(64, 0.5), "k"), true);
let seed = 1;
const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 2 ** 32);
let collide = 0;
for (let trial = 0; trial < 500; trial++) {
  const a = new Float32Array(2 * (10 + Math.floor(rnd() * 200)));
  for (let i = 0; i < a.length; i++) a[i] = rnd() * 2 - 1;
  const b = a.slice();
  const i = Math.floor(rnd() * b.length);
  const j = (i + 1 + Math.floor(rnd() * (b.length - 1))) % b.length;
  if (trial % 2) { b[i] = -b[i]; b[j] = -b[j]; } else { b[i] = b[i] === 0 ? 1 : 0; }
  if (hashF32(a, "k") === hashF32(b, "k")) collide++;
}
same("乱数の配列で 2 か所の符号・1 か所の値を変えても衝突しない", collide, 0);

// ── 2. 縦の一筆の幅 ─────────────────────────
for (const bpm of [80, 120]) {
  const spt = secPerTick(bpm);
  const frameTicks = 1 / FPS / spt;
  const minW = 3 * frameTicks;
  /** phraseParams と同じく、フレームの頭で曲線を読んだとき、どこかのフレームが th 以上か */
  const heard = (c: Curve, th: number) => {
    for (let f = 0; f < 2000; f++) if (evalCurve(c, f / FPS / spt, 0) >= th) return true;
    return false;
  };
  for (const off of [0, 0.3, 0.7]) {
    const t = 480 + off * frameTicks;
    const c: Curve = { pts: [[0, 0], [960, 0]], interp: "cubic" };
    const s = new PenStroke(c, 0, 2 * frameTicks);
    s.add(t, 0, false);
    s.add(t, 1);                        // 縦にまっすぐ上へ
    s.finish(minW);
    same(`縦の一筆が聞こえる（${bpm} BPM、ずれ ${off}）`, heard(c, 0.9), true);
    const c2: Curve = { pts: [[0, 0], [960, 0]], interp: "cubic" };
    lineRange(c2, t, 0, t, 1, 2 * frameTicks, 0, minW);
    same(`縦の直線が聞こえる（${bpm} BPM、ずれ ${off}）`, heard(c2, 0.9), true);
  }
  // 幅のある一筆は今までどおり（広げない）
  const c3: Curve = { pts: [], interp: "cubic" };
  const s3 = new PenStroke(c3, 0, 2 * frameTicks);
  s3.add(100, 0.5, false);
  s3.add(400, 0.5);
  s3.finish(minW);
  same(`幅のある一筆は広げない（${bpm} BPM）`, [c3.pts[0][0] <= 100, c3.pts[c3.pts.length - 1][0] >= 400,
    evalCurve(c3, 250, 0) > 0.49 && evalCurve(c3, 250, 0) < 0.51], [true, true, true]);
}

// ── 3. 速い山を取りこぼさない ───────────────
{
  const frameTicks = 1 / FPS / secPerTick(80);
  const c: Curve = { pts: [[0, 0], [960, 0]], interp: "cubic" };
  const s = new PenStroke(c, 0, 2 * frameTicks);   // 鉛筆の間隔は 2 フレームより粗くしない（paramview.ts）
  s.add(110, 0, false);
  s.add(116.7, 1);
  s.add(123.3, 0);
  same("速い山の頂点が残る", Math.max(...c.pts.map(([, v]) => v)) >= 0.9, true);
}

done();
