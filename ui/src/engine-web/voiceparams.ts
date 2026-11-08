// ボイスパラメータのうち、ボコーダの出力（調波・雑音の波形）より後で掛けるもの（engine/voiceparams.py と同じ）。
// ボコーダを回し直さずに済むので、スライダーを動かしても混ぜ直すだけで済む。
//   spectral    テンション・ジェンダー・響き（STFT、窓 1024・hop 256 で周波数ごとの利得を掛ける）
//   branchMix   ブレシネス・有声度・子音の強さ（枝ごとの音量）
//   finish      うなり（hifisampler の growl）・音量・軽いリミッタ
// Python（numpy・scipy）の型の扱いに合わせる: float32 の配列と Python の数の演算は float32 のまま（NEP 50）、
// float32 と float64 の配列の演算は float64。float32 で計算される所は Math.fround で丸める。
import { RealFFT, hannPeriodic } from "./melspec";

const HOP = 256;
const SR = 44100;
const fr32 = Math.fround;

/** パラメータの既定値（何もしない値）。voiceparams.NEUTRAL */
export const NEUTRAL: Record<string, number> = {
  tension: 0.0, breathiness: 0.0, gender: 0.0, voicing: 1.0,
  loudness: 0.0, consonant: 0.0, resonance: 0.0, growl: 0.0,
  articulation: 0.0, // 滑舌（子音の凹みの既定の深さの倍率。0 = 凹み無し）。音響モデルの前で効く（core.ts の凹み）
};
/** 周波数ごとの利得で掛けるもの。voiceparams.SPECTRAL_KEYS */
export const SPECTRAL_KEYS = ["tension", "gender", "resonance"];
const CONSONANT_DB = 12.0;
const TENSION_BETA = 0.6; // テンション ±1 で ±3.6 dB/オクターブ
const TENSION_PIVOT = 1000.0; // 傾きの支点 [Hz]
const TENSION_FMIN = 100.0; // これより低い周波数は傾きを延ばさない
const GENDER_OCT = 0.3; // ジェンダー ±1 で 4 kHz までの包絡を 2^±0.3 倍
const GENDER_SPLIT = 4000.0;
const RESONANCE_DB = 10.0;
const RES_LO = 2000.0;
const RES_HI = 4000.0;
const N_FFT = 1024; // ボコーダのフィルタと同じ（包絡の bin と揃う）

export type Params = Record<string, ArrayLike<number> | undefined>;

/** voiceparams.frame_param: フレームごとの値 [T]（float32。無ければ既定値、短ければ最後の値を延ばす） */
export function frameParam(params: Params, key: string, T: number): Float32Array {
  const a = params[key];
  const out = new Float32Array(T);
  if (a == null || a.length === 0) return out.fill(NEUTRAL[key]);
  const n = Math.min(a.length, T);
  for (let i = 0; i < n; i++) out[i] = a[i];
  for (let i = n; i < T; i++) out[i] = out[n - 1];
  return out;
}

/** voiceparams.is_neutral: どのキーも無いか、既定値と np.allclose（atol 1e-8・rtol 1e-5）で等しい */
export function isNeutral(params: Params, keys: string[]): boolean {
  return keys.every((k) => {
    const a = params[k];
    if (a == null) return true;
    const b = NEUTRAL[k];
    for (let i = 0; i < a.length; i++) {
      if (!(Math.abs(fr32(a[i]) - b) <= 1e-8 + 1e-5 * Math.abs(b))) return false;
    }
    return true;
  });
}

/** voiceparams._to_samples: フレームの値 → サンプルの値（3 フレームの移動平均の後、フレームの中心を結ぶ線形補間）。float32 で返す。
 *  画面側がブレスに「音量」の曲線を掛けるときも同じ式を使う（breath.ts） */
export function toSamples(g: ArrayLike<number>, n: number): Float32Array {
  let v: Float64Array;
  if (g.length > 2) {
    // np.convolve(np.pad(g, 1, "edge"), ones(3)/3, "valid")（float64）
    const L = g.length;
    const third = 1 / 3;
    v = new Float64Array(L);
    for (let i = 0; i < L; i++) {
      const a = g[Math.max(i - 1, 0)];
      const b = g[i];
      const c = g[Math.min(i + 1, L - 1)];
      v[i] = a * third + b * third + c * third;
    }
  } else {
    v = Float64Array.from(g as ArrayLike<number>);
  }
  // np.interp(arange(n), (arange(len)+0.5)·hop, v): 端の外は端の値
  const out = new Float32Array(n);
  const L = v.length;
  if (L === 0) return out;
  const c0 = 0.5 * HOP;
  const cl = (L - 1 + 0.5) * HOP;
  for (let i = 0; i < n; i++) {
    if (L === 1 || i <= c0) out[i] = v[0];
    else if (i >= cl) out[i] = v[L - 1];
    else {
      const j = Math.floor((i - c0) / HOP);
      const xj = (j + 0.5) * HOP;
      const slope = (v[j + 1] - v[j]) / ((j + 1 + 0.5) * HOP - xj);
      out[i] = i === xj ? v[j] : slope * (i - xj) + v[j];
    }
  }
  return out;
}

const melScale = (f: number): number => 2595.0 * Math.log10(1.0 + f / 700.0);

// 周波数の bin ごとの表（voiceparams の _F・_BUMP・_LNF）
const NH = N_FFT / 2 + 1;
const F = new Float64Array(NH);
const BUMP = new Float64Array(NH);
const LNF = new Float64Array(NH);
const MELF = new Float64Array(NH);
for (let k = 0; k < NH; k++) {
  F[k] = (k * SR) / N_FFT;
  const f = F[k];
  const b = f > RES_LO && f < RES_HI ? 0.5 - 0.5 * Math.cos((2 * Math.PI * (f - RES_LO)) / (RES_HI - RES_LO)) : 0.0;
  BUMP[k] = (b * RESONANCE_DB * Math.log(10)) / 20; // 自然対数の振幅
  LNF[k] = Math.log(Math.max(f, TENSION_FMIN) / TENSION_PIVOT);
  MELF[k] = melScale(f);
}
const M4 = melScale(GENDER_SPLIT);
const MN = melScale(SR / 2);

/** voiceparams._at_stft の添字: ボコーダのフレーム（中心 (t+0.5)·hop）→ STFT のフレーム i（中心 i·hop）の線形補間 */
function atStft(i: number, len: number): [number, number, number] {
  const pos = Math.min(Math.max(i - 0.5, 0.0), len - 1.0);
  const i0 = Math.floor(pos);
  const i1 = Math.min(i0 + 1, len - 1);
  return [i0, i1, pos - i0];
}
function atStftVec(x: ArrayLike<number>, nSt: number): Float64Array {
  const out = new Float64Array(nSt);
  for (let i = 0; i < nSt; i++) {
    const [i0, i1, w] = atStft(i, x.length);
    out[i] = x[i0] * (1 - w) + x[i1] * w;
  }
  return out;
}

let FFT1024: RealFFT | null = null;
let WIN1024: Float64Array | null = null;

/** scipy.signal.stft(x, window="hann", nperseg=1024, noverlap=768, boundary="zeros", padded=True) の各フレームを返す。
 * scipy は float32 の入力では窓を float32 に丸めて掛け（frames = fl32(fl32(win)·x)）、1/Σwin で割る */
function stftFrames(x: Float32Array): { re: Float64Array; im: Float64Array; nSt: number } {
  FFT1024 ??= new RealFFT(N_FFT);
  WIN1024 ??= hannPeriodic(N_FFT);
  const win = WIN1024;
  const half = N_FFT / 2;
  const L0 = x.length + 2 * half;
  const nadd = (((-(L0 - N_FFT)) % HOP) + HOP) % HOP % N_FFT;
  const L = L0 + nadd;
  const nSt = 1 + Math.floor((L - N_FFT) / HOP);
  let wsum = 0;
  const w32 = new Float64Array(N_FFT);
  for (let j = 0; j < N_FFT; j++) {
    w32[j] = fr32(win[j]);
    wsum += w32[j];
  }
  const scale = 1 / fr32(wsum);
  const re = new Float64Array(nSt * NH);
  const im = new Float64Array(nSt * NH);
  const frame = new Float64Array(N_FFT);
  const r = new Float64Array(NH);
  const q = new Float64Array(NH);
  for (let i = 0; i < nSt; i++) {
    const off = i * HOP - half; // 元の信号での頭
    for (let j = 0; j < N_FFT; j++) {
      const s = off + j;
      frame[j] = s >= 0 && s < x.length ? fr32(w32[j] * x[s]) : 0;
    }
    FFT1024.forward(frame, r, q);
    for (let k = 0; k < NH; k++) {
      re[i * NH + k] = r[k] * scale;
      im[i * NH + k] = q[k] * scale;
    }
  }
  return { re, im, nSt };
}

/** scipy.signal.istft(·, window="hann", nperseg=1024, noverlap=768, boundary=True) の重ね合わせ。
 * gain(i, lnOut) がフレーム i の ln 利得 [513] を書き込む。戻り値は長さ n に詰めた float32 */
function istftWithGain(re: Float64Array, im: Float64Array, nSt: number, n: number,
                       gain: (i: number, ln: Float64Array) => void): Float32Array {
  const fft = FFT1024!;
  const win = WIN1024!;
  const half = N_FFT / 2;
  let S = 0;
  for (let j = 0; j < N_FFT; j++) S += win[j];
  const outLen = N_FFT + (nSt - 1) * HOP;
  const x = new Float64Array(outLen);
  const norm = new Float64Array(outLen);
  const ln = new Float64Array(NH);
  const r = new Float64Array(NH);
  const q = new Float64Array(NH);
  const xs = new Float64Array(N_FFT);
  for (let i = 0; i < nSt; i++) {
    gain(i, ln);
    for (let k = 0; k < NH; k++) {
      const e = Math.exp(ln[k]);
      r[k] = re[i * NH + k] * e;
      q[k] = im[i * NH + k] * e;
    }
    fft.inverse(r, q, xs);
    const off = i * HOP;
    for (let j = 0; j < N_FFT; j++) {
      x[off + j] += xs[j] * S * win[j];
      norm[off + j] += win[j] * win[j];
    }
  }
  // 両端の nperseg//2 を落とし、窓の重なりで割る（norm が小さい所は割らない）。長さ n に詰める
  const out = new Float32Array(n);
  const m = Math.min(n, outLen - 2 * half);
  for (let t = 0; t < m; t++) {
    const nv = norm[t + half];
    out[t] = x[t + half] / (nv > 1e-10 ? nv : 1.0);
  }
  return out;
}

/** voiceparams.spectral: テンション・ジェンダー・響きを、調波・雑音の波形に周波数ごとの利得として掛ける。
 * envH・envN: ボコーダのフィルタの滑らかな包絡 [T', 513]（行優先・自然対数の振幅。float16 に丸めたもの） */
export function spectral(harm: Float32Array, noise: Float32Array, envH: Float32Array | null, envN: Float32Array | null,
                         params: Params, T: number): [Float32Array, Float32Array] {
  if (envH == null || envN == null || isNeutral(params, SPECTRAL_KEYS)) return [harm, noise];
  const n = harm.length;
  const H = stftFrames(harm);
  const Z = stftFrames(noise);
  const nSt = H.nSt;
  const t = atStftVec(frameParam(params, "tension", T), nSt);
  const g = atStftVec(frameParam(params, "gender", T), nSt);
  const r = atStftVec(frameParam(params, "resonance", T), nSt);
  const anyG = g.some((v) => v !== 0);
  const anyR = r.some((v) => v !== 0);
  const anyT = t.some((v) => v !== 0);
  const rowsH = envH.length / NH;
  const rowsN = envN.length / NH;
  const src = new Float64Array(NH);

  // ジェンダー: 出力の各 bin が読みに行く元の bin（_gender_src）。0〜4 kHz は α 倍、その上は mel 軸で 22.05 kHz へ直線でつなぐ
  const genderSrc = (alpha: number): void => {
    const mA4 = melScale(alpha * GENDER_SPLIT);
    for (let k = 0; k < NH; k++) {
      const hzUp = 700.0 * (10.0 ** ((mA4 + ((MELF[k] - M4) * (MN - mA4)) / (MN - M4)) / 2595.0) - 1.0);
      const s = (F[k] <= GENDER_SPLIT ? F[k] * alpha : hzUp) * (N_FFT / SR);
      src[k] = Math.min(Math.max(s, 0.0), NH - 1.0);
    }
  };
  // 包絡を伸び縮みさせたもの − 包絡（_warp(env, src) − env。env は STFT のフレームへ補間したもの）
  const warpDiff = (env: Float32Array, rows: number, i: number, out: Float64Array): void => {
    const [i0, i1, w] = atStft(i, rows);
    const a0 = i0 * NH;
    const a1 = i1 * NH;
    const e = (k: number): number => env[a0 + k] * (1 - w) + env[a1 + k] * w;
    for (let k = 0; k < NH; k++) {
      const j = Math.min(Math.floor(src[k]), NH - 2);
      const f = src[k] - j;
      const a = e(j);
      out[k] = a + (e(j + 1) - a) * f - e(k);
    }
  };

  // 調波の ln 利得（フレーム i）
  const lnHarm = (i: number, ln: Float64Array): void => {
    ln.fill(0);
    if (anyG) {
      genderSrc(2.0 ** (GENDER_OCT * g[i]));
      warpDiff(envH, rowsH, i, ln);
    }
    if (anyR) for (let k = 0; k < NH; k++) ln[k] += r[i] * BUMP[k];
    if (anyT) {
      // 傾けた後の調波のエネルギーがフレームごとに変わらないよう戻す
      const bt = TENSION_BETA * t[i];
      let e0 = 0;
      let e1 = 0;
      for (let k = 0; k < NH; k++) {
        const hr = H.re[i * NH + k];
        const hi = H.im[i * NH + k];
        const p = (hr * hr + hi * hi) * Math.exp(2 * ln[k]);
        e0 += p;
        e1 += p * Math.exp(2 * (bt * LNF[k]));
      }
      const norm = 0.5 * Math.log(Math.max(e0, 1e-20) / Math.max(e1, 1e-20));
      const nc = Math.min(Math.max(norm, -3.0), 3.0);
      for (let k = 0; k < NH; k++) ln[k] += bt * LNF[k] + nc;
    }
    for (let k = 0; k < NH; k++) ln[k] = Math.min(Math.max(ln[k], -8.0), 8.0);
  };
  // 雑音の ln 利得（ジェンダーだけ）。調波と同じ α で src を作り直す（調波の istft を先に全部回すため）
  const lnNoise = (i: number, ln: Float64Array): void => {
    if (anyG) {
      genderSrc(2.0 ** (GENDER_OCT * g[i]));
      warpDiff(envN, rowsN, i, ln);
      for (let k = 0; k < NH; k++) ln[k] = Math.min(Math.max(ln[k], -8.0), 8.0);
    } else ln.fill(0);
  };
  const h2 = istftWithGain(H.re, H.im, nSt, n, lnHarm);
  const z2 = istftWithGain(Z.re, Z.im, Z.nSt, n, lnNoise);
  return [h2, z2];
}

/** voiceparams.branch_mix: 調波と雑音を、ブレシネス・有声度・子音の強さで重みを付けて足す。
 * cons: 子音の音素のフレーム [T]（無ければ無声のフレームを子音とみなす） */
export function branchMix(harm: Float32Array, noise: Float32Array, uvHard: ArrayLike<number>, params: Params,
                          cons: ArrayLike<number> | null = null): Float32Array {
  const T = uvHard.length;
  const n = Math.min(harm.length, noise.length, T * HOP);
  const b = frameParam(params, "breathiness", T);
  const v = frameParam(params, "voicing", T);
  const c = frameParam(params, "consonant", T);
  const nf = Math.floor(n / HOP);
  const gh = new Float64Array(T);
  const gn = new Float32Array(T);
  for (let f = 0; f < T; f++) {
    // フレームごとのエネルギー（float64 の平均）
    let eh = 0;
    let en = 0;
    if (f < nf) {
      for (let j = f * HOP; j < (f + 1) * HOP; j++) {
        eh += harm[j] * harm[j];
        en += noise[j] * noise[j];
      }
      eh /= HOP;
      en /= HOP;
    }
    const voiced = uvHard[f] > 0.5;
    // 10 ** (9b/20) は float32 の配列の演算（float32 のまま）
    const gnf = voiced ? fr32(10 ** fr32(fr32(9.0 * b[f]) / 20.0)) : 1.0;
    // 有声の所: 雑音を増やした分だけ調波を下げ、全体のエネルギーを保つ
    let h = Math.sqrt(Math.max(0.0, eh + en - fr32(gnf * gnf) * en) / Math.max(eh, 1e-12));
    h = voiced ? Math.min(Math.max(h, 0.0), 2.0) : 1.0;
    h = h * Math.min(Math.max(v[f], 0.0), 1.0);
    // 子音の強さ: 子音の音素のフレーム（有声の子音も含む）を両方の枝ごと ±CONSONANT_DB
    const cm = cons == null ? !voiced : Boolean(cons[f]);
    const gc = cm ? fr32(10 ** fr32(fr32(CONSONANT_DB * c[f]) / 20.0)) : 1.0;
    gh[f] = h * gc;
    gn[f] = fr32(gnf * gc);
  }
  const sh = toSamples(gh, n);
  const sn = toSamples(gn, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = harm[i] * sh[i] + noise[i] * sn[i];
  return out;
}

// ── うなり（hifisampler の util/growl.py と同じ） ──

type Sos = number[][]; // [b0, b1, b2, 1, a1, a2] の並び

// 複素数の小さな道具（scipy の zpk の計算を同じ順で行う）
type C = [number, number];
const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
/** numpy の複素数の割り算（Smith の方法） */
function cdiv(a: C, b: C): C {
  const [ar, ai] = a;
  const [br, bi] = b;
  if (Math.abs(br) >= Math.abs(bi)) {
    const rat = bi / br;
    const scl = 1.0 / (br + bi * rat);
    return [(ar + ai * rat) * scl, (ai - ar * rat) * scl];
  }
  const rat = br / bi;
  const scl = 1.0 / (bi + br * rat);
  return [(ar * rat + ai) * scl, (ai * rat - ar) * scl];
}

/** scipy.signal.butter(N, Wn, "high", output="sos")（N は偶数）。buttap → lp2hp_zpk → bilinear_zpk → zpk2sos("nearest") */
export function butterHighSos(N: number, Wn: number): Sos {
  // buttap: p = −exp(iπm/(2N))、m = −N+1, −N+3, …, N−1
  const p: C[] = [];
  for (let m = -N + 1; m < N; m += 2) {
    const th = (Math.PI * m) * (1 / (2 * N));
    p.push([-Math.cos(th), -Math.sin(th)]);
  }
  const fs = 2.0;
  const warped = 2 * fs * Math.tan((Math.PI * Wn) / fs);
  // lp2hp_zpk: 極を wo/p へ、零点は原点に N 個。k = real(1 / prod(−p))
  const php = p.map((x) => cdiv([warped, 0], x));
  let pr: C = [1, 0];
  for (const x of p) pr = cmul(pr, [-x[0], -x[1]]);
  const khp = cdiv([1, 0], pr)[0];
  // bilinear_zpk（fs2 = 4）: 零点 (4+0)/(4−0) = 1、極 (4+p)/(4−p)、k = k·real(prod(4 − z) / prod(4 − p))
  const fs2 = 2.0 * fs;
  const pz = php.map((x) => cdiv([fs2 + x[0], x[1]], [fs2 - x[0], -x[1]]));
  let pd: C = [1, 0];
  for (const x of php) pd = cmul(pd, [fs2 - x[0], -x[1]]);
  const kz = khp * cdiv([fs2 ** N, 0], pd)[0];
  // zpk2sos: 共役の対の片方（虚部が正）を選び、単位円に近い極ほど後ろの段へ。零点はすべて z = 1（b = [1, −2, 1]）
  const pos = pz.filter((x) => x[1] > 0);
  const neg = pz.filter((x) => x[1] < 0);
  pos.sort((a, b) => a[0] - b[0]);
  neg.sort((a, b) => a[0] - b[0]);
  // _cplxreal: 正の虚部の極と、負の虚部の極の共役を平均する
  let pairs: C[] = pos.map((x, i) => [(x[0] + neg[i][0]) / 2, (x[1] - neg[i][1]) / 2]);
  const nSec = pairs.length;
  const sos: Sos = new Array(nSec);
  for (let si = nSec - 1; si >= 0; si--) {
    let best = 0;
    for (let j = 1; j < pairs.length; j++) {
      if (Math.abs(1 - Math.hypot(...pairs[j])) < Math.abs(1 - Math.hypot(...pairs[best]))) best = j;
    }
    const [r, i] = pairs[best];
    pairs = pairs.filter((_, j) => j !== best);
    sos[si] = [1, -2, 1, 1, -(r + r), r * r + i * i];
  }
  for (let j = 0; j < 3; j++) sos[0][j] *= kz;
  return sos;
}

/** scipy.signal.sosfilt（初期状態 0、直接形 II 転置の段を縦につなぐ） */
function sosfilt(sos: Sos, x: Float64Array): Float64Array {
  const ns = sos.length;
  const z0 = new Float64Array(ns);
  const z1 = new Float64Array(ns);
  const y = new Float64Array(x.length);
  for (let n = 0; n < x.length; n++) {
    let xc = x[n];
    for (let s = 0; s < ns; s++) {
      const c = sos[s];
      const xn = c[0] * xc + z0[s];
      z0[s] = c[1] * xc - c[4] * xn + z1[s];
      z1[s] = c[2] * xc - c[5] * xn;
      xc = xn;
    }
    y[n] = xc;
  }
  return y;
}

let SOS_GROWL: Sos | null = null;
let SOS_DRIFT: Sos | null = null;

/** voiceparams.growl（hifisampler の growl と同じ）: 400 Hz より上の帯域を 80 Hz の矩形の LFO で ±100·strength セントだけ
 * 時間方向にゆがめ、その帯域の RMS を戻す。strength はサンプルごと */
function growl(audio: Float32Array, strength: Float32Array): Float32Array {
  const frequency = 80.0;
  const freqLow = 400.0;
  if (strength.every((s) => s === 0)) return audio;
  const N = audio.length;
  SOS_GROWL ??= butterHighSos(4, Math.min(Math.max(freqLow / (SR / 2), 0.01), 0.99));
  SOS_DRIFT ??= butterHighSos(2, 20.0 / (SR / 2));
  const x = Float64Array.from(audio);
  const band = sosfilt(SOS_GROWL, x);
  // 矩形の LFO（scipy.signal.square、duty 0.5）: mod(t, 2π) < π で 1、それ以外は −1
  const c = 2 * Math.PI * frequency;
  const ratio = new Float64Array(N);
  let rsum = 0;
  for (let i = 0; i < N; i++) {
    const t = (c * i) / SR;
    const lfo = t % (2 * Math.PI) < 1 * Math.PI ? 1 : -1;
    ratio[i] = 2 ** ((lfo * strength[i] * 100.0) / 1200.0);
    rsum += ratio[i];
  }
  const mean = rsum / N;
  let drift = new Float64Array(N);
  let cs = 0;
  for (let i = 0; i < N; i++) {
    cs += ratio[i];
    drift[i] = cs - ratio[0] - i * mean;
  }
  if (N > 100) drift = sosfilt(SOS_DRIFT, drift);
  // mod = np.interp(clip(n + drift, 0, N−1), n, band)
  const mod = new Float64Array(N);
  let s0 = 0;
  let s1 = 0;
  for (let i = 0; i < N; i++) {
    const q = Math.min(Math.max(i + drift[i], 0), N - 1);
    let v: number;
    if (q >= N - 1) v = band[N - 1];
    else {
      const j = Math.floor(q);
      v = q === j ? band[j] : (band[j + 1] - band[j]) * (q - j) + band[j];
    }
    mod[i] = v;
    s0 += band[i] * band[i];
    s1 += v * v;
  }
  const r0 = Math.sqrt(s0 / N);
  const r1 = Math.sqrt(s1 / N);
  const out = new Float32Array(N);
  const sc = r1 > 1e-10 ? r0 / r1 : 1;
  for (let i = 0; i < N; i++) out[i] = x[i] - band[i] + (r1 > 1e-10 ? mod[i] * sc : mod[i]);
  return out;
}

/** voiceparams.finish: うなりと音量を掛け、軽いリミッタを通す（0.9 を超える所だけを tanh で丸め、1.0 を超えない） */
export function finish(wav: Float32Array, params: Params, T: number): Float32Array {
  const n = wav.length;
  let w = wav;
  const g = frameParam(params, "growl", T);
  if (g.some((x) => x !== 0)) {
    const gc = new Float32Array(T);
    for (let i = 0; i < T; i++) gc[i] = Math.min(Math.max(g[i], 0.0), 1.0);
    w = growl(w, toSamples(gc, n));
  }
  const db = frameParam(params, "loudness", T);
  if (db.some((x) => x !== 0)) {
    // wav · 10 ** (dB/20)（float32 の配列の演算なので float32 のまま）
    const s = toSamples(db, n);
    const o = new Float32Array(n);
    for (let i = 0; i < n; i++) o[i] = w[i] * fr32(10 ** fr32(s[i] / 20.0));
    w = o;
  }
  const t9 = fr32(0.9);
  const t1 = fr32(0.1);
  let out: Float32Array | null = null;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(w[i]);
    if (a > t9) {
      out ??= Float32Array.from(w);
      const y = fr32(t9 + fr32(t1 * fr32(Math.tanh(fr32(fr32(a - t9) / t1)))));
      out[i] = Math.sign(w[i]) * y;
    }
  }
  return out ?? w; // Python と同じく、何も掛けなければ入力をそのまま返す
}
