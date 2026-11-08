// F0 のモデル（f0.onnx）の前後の処理。Python の engine/pipeline.py の PrettyPitchLeapSinger._f0 と、
// analyze の最後（plain_log2・vib_log2・uv・base_log2 を作る所）と同じ計算をする。
// 中で呼んでいる PrettyPitch の関数（svs/render.py・infer.py・prettypitch/baseline.py・correction.py・pitch.py）も
// ここに写してある。scipy・numpy の細かい振る舞い（端の扱い・float32 の丸め・偶数への丸め・足し算の順番）まで合わせる。
// モデルを走らせる所（ONNX）はここに無い: prepareF0 → f0Feeds → （呼ぶ側がモデルを走らせる）→ finishF0 の順に使う。
import type { Feeds, PhonemeSequence, VoicebankWeb } from "./types";
import { f32, i64 } from "./types";

// ── 定数（Python のモジュールにあるもの） ──

/** prettypitch/baseline.py の LOG2_440 = float(np.log2(440.0)) */
const LOG2_440 = 8.78135971352466;
/** prettypitch/baseline.py の GAUSSIAN_SIGMA_SEC（f0.json に baseline_sigma_sec が無いときの値） */
const GAUSSIAN_SIGMA_SEC = 0.02;
// prettypitch/correction.py の既定値
const TUKEY_ALPHA = 0.5; // _TUKEY_ALPHA（第 1 段の端の重み）
const SG_WINDOW_MS = 55.0; // _SG_WINDOW_MS
const SG_POLY_ORDER = 3; // _SG_POLY_ORDER
const TARGET_SIGMA_ST = 0.5; // _TARGET_SIGMA_ST
const SMOOTH_SIGMA_MS = 30.0; // _SMOOTH_SIGMA_MS（第 1 段の補正量を丸める幅）
const HANN_SMOOTH_SIGMA_MS = 11.0; // apply_hann_pitch_correction の smooth_sigma_ms の既定
// infer.py の correct_f0 の既定値（pipeline の _f0 はこれをそのまま使う）
const CONS_WEIGHT = 0.0;
const HANN_ALPHA = 0.8;
const HANN_LEAD_TAPER = 0.6;
const HANN_TRAIL_TAPER = 0.1;

// ── 型 ──

/** モデルの前までに作ったもの（prepareF0 の戻り値）。T・phDurs から呼ぶ側が雑音 [1, n_noise, 2, T] を作る */
export interface F0Prep {
  phStrs: string[]; // 音素の名前（_read_score の ph_strs）
  phIds: Int32Array; // f0.json の phonemes での番号（tokens）
  durSec: Float32Array; // 音素の長さ（秒、float32）
  values: Float64Array; // 音素ごとの MIDI（音高の無い所は NaN）
  T: number; // フレーム数（_frame_grid の T = phDurs の合計）
  phDurs: Int32Array; // 音素ごとのフレーム数（各 1 以上）
  baseline: Float32Array; // 抑揚の土台（log2 Hz）[T]
  pitchMidi: Float32Array; // モデルへの音高（核の値を後ろへ持ち回ったもの）[L]
  styleMix: Float32Array; // [n_styles]（先頭だけ 1）
  spk: number;
  nNoise: number;
}

/** F0 の最後（finishF0 の戻り値）。analyze が PhraseAnalysis に入れる 4 本と、照らし合わせ用の途中の値 */
export interface F0Result {
  T: number;
  plainLog2: Float32Array; // 補正後の抑揚の線（log2 Hz）
  vibLog2: Float32Array; // ビブラート（log2 のずれ）
  uv: Float32Array; // 有声 = 1
  baseLog2: Float32Array; // 抑揚の土台（log2 Hz）
  // ↓ 途中の値（試験用）
  phonemesPf: string[];
  notePitchesPf: Float64Array;
  noteBoundariesPf: Uint8Array;
  frameIsPau: Uint8Array;
  plainDelta: Float32Array;
  plainHzBeforeCorrection: Float32Array;
  plainHz: Float32Array; // 補正後（correct_f0 の plain_hz）
  f0Hz: Float32Array; // 補正後の plain_hz × 2^vib（correct_f0 の f0_hz）
}

// ── numpy・scipy の振る舞いを写した小道具 ──

const fr32 = Math.fround;

/** np.round・Python の round と同じ、ちょうど半分のときは偶数へ丸める */
function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * numpy の足し算の reduce（np.sum）と同じ順番で足す（pairwise_sum。8 本に分けて足し、128 個を超えると半分に割る）。
 * 初期値 0 に全部の和を足す形も numpy と同じ。single=true なら float32 で足す（float32 の配列の np.sum）
 */
function npSum(a: ArrayLike<number>, single = false, off = 0, n = a.length - off): number {
  const r = single ? fr32 : (x: number) => x;
  const pw = (o: number, m: number): number => {
    if (m < 8) {
      let s = r(0);
      for (let i = 0; i < m; i++) s = r(s + a[o + i]);
      return s;
    }
    if (m <= 128) {
      const acc = [a[o], a[o + 1], a[o + 2], a[o + 3], a[o + 4], a[o + 5], a[o + 6], a[o + 7]].map((x) => r(x));
      let i = 8;
      for (; i < m - (m % 8); i += 8) for (let j = 0; j < 8; j++) acc[j] = r(acc[j] + a[o + i + j]);
      let s = r(r(r(acc[0] + acc[1]) + r(acc[2] + acc[3])) + r(r(acc[4] + acc[5]) + r(acc[6] + acc[7])));
      for (; i < m; i++) s = r(s + a[o + i]);
      return s;
    }
    let n2 = Math.floor(m / 2);
    n2 -= n2 % 8;
    return r(pw(o, n2) + pw(o + n2, m - n2));
  };
  return r(0 + pw(off, n));
}

/** scipy.ndimage の mode='reflect'（d c b a | a b c d | d c b a）での番号。長く外へ出ても 2n の周期で折り返す */
function reflectIndex(i: number, n: number): number {
  if (n === 1) return 0;
  const p = 2 * n;
  const m = ((i % p) + p) % p;
  return m < n ? m : p - 1 - m;
}

/**
 * scipy.ndimage.gaussian_filter1d（order 0・mode 'reflect'・truncate 4.0）。
 * 窓は _gaussian_kernel1d と同じ式で作り、畳み込みは NI_Correlate1D の対称な窓の道（遠い方から足す）と同じ順で足す
 */
function gaussianFilter1d(x: Float64Array, sigma: number): Float64Array {
  const n = x.length;
  const lw = Math.trunc(4.0 * sigma + 0.5);
  const c = -0.5 / (sigma * sigma);
  const phi = new Float64Array(2 * lw + 1);
  for (let i = 0; i <= 2 * lw; i++) {
    const k = i - lw;
    phi[i] = Math.exp(c * (k * k));
  }
  const s = npSum(phi);
  for (let i = 0; i < phi.length; i++) phi[i] = phi[i] / s;
  const out = new Float64Array(n);
  if (n === 0) return out;
  for (let i = 0; i < n; i++) {
    let acc = x[i] * phi[lw];
    for (let j = lw; j >= 1; j--) acc += (x[reflectIndex(i - j, n)] + x[reflectIndex(i + j, n)]) * phi[lw - j];
    out[i] = acc;
  }
  return out;
}

/**
 * W 点（0..W-1）の上で poly 次以下の多項式へ最小二乗で当てはめる射影の行列 H [W][W]（y = H x）。
 * savgol_coeffs（中央の行）と savgol_filter の端の polyfit（端の行）の両方がこれの行になる。
 * 数値の安定のため、横軸を中央へ寄せ、修正グラム・シュミット（2 回）で正規直交の基底を作る
 */
function polyProjection(W: number, poly: number): Float64Array[] {
  const K = poly + 1;
  const q: Float64Array[] = [];
  const c0 = (W - 1) / 2;
  for (let k = 0; k < K; k++) {
    const v = new Float64Array(W);
    for (let i = 0; i < W; i++) v[i] = Math.pow(i - c0, k);
    for (let pass = 0; pass < 2; pass++) {
      for (const u of q) {
        let d = 0;
        for (let i = 0; i < W; i++) d += u[i] * v[i];
        for (let i = 0; i < W; i++) v[i] -= d * u[i];
      }
    }
    let nr = 0;
    for (let i = 0; i < W; i++) nr += v[i] * v[i];
    nr = Math.sqrt(nr);
    for (let i = 0; i < W; i++) v[i] /= nr;
    q.push(v);
  }
  const H: Float64Array[] = [];
  for (let i = 0; i < W; i++) {
    const row = new Float64Array(W);
    for (let j = 0; j < W; j++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += q[k][i] * q[k][j];
      row[j] = s;
    }
    H.push(row);
  }
  return H;
}

/**
 * scipy.signal.savgol_filter(x, win, poly)（deriv 0・mode 'interp'）。
 * 中は窓の係数で畳み込み（mode 'constant'）、両端の win//2 点は端の win 点へ当てはめた多項式の値で置き換える
 */
function savgolFilterInterp(x: Float64Array, win: number, poly: number): Float64Array {
  const n = x.length;
  const h = Math.floor(win / 2);
  const H = polyProjection(win, poly);
  const w = H[h]; // 中央の行 = savgol_coeffs（左右対称）
  const y = new Float64Array(n);
  const at = (i: number) => (i >= 0 && i < n ? x[i] : 0);
  for (let i = 0; i < n; i++) {
    let acc = x[i] * w[h];
    for (let j = h; j >= 1; j--) acc += (at(i - j) + at(i + j)) * w[h - j];
    y[i] = acc;
  }
  // 端（_fit_edges_polyfit）: 頭は x[0..win-1]、尻は x[n-win..n-1] へ当てはめる
  for (let i = 0; i < h; i++) {
    let s = 0;
    for (let j = 0; j < win; j++) s += H[i][j] * x[j];
    y[i] = s;
  }
  for (let k = 0; k < h; k++) {
    const r = H[win - h + k];
    let s = 0;
    for (let j = 0; j < win; j++) s += r[j] * x[n - win + j];
    y[n - h + k] = s;
  }
  return y;
}

/** scipy.signal.windows.tukey(M, alpha)（sym=True、0 < alpha < 1 のとき） */
function tukeyWindow(M: number, alpha: number): Float64Array {
  if (M <= 1) return new Float64Array(M).fill(1);
  if (alpha <= 0) return new Float64Array(M).fill(1);
  if (alpha >= 1) throw new Error("tukey: alpha >= 1（hann）は使っていない");
  const w = new Float64Array(M);
  const width = Math.floor((alpha * (M - 1)) / 2.0);
  for (let i = 0; i < M; i++) {
    if (i < width + 1) w[i] = 0.5 * (1 + Math.cos(Math.PI * (-1 + (2.0 * i) / alpha / (M - 1))));
    else if (i < M - width - 1) w[i] = 1.0;
    else w[i] = 0.5 * (1 + Math.cos(Math.PI * (-2.0 / alpha + 1 + (2.0 * i) / alpha / (M - 1))));
  }
  return w;
}

/** np.linspace(start, stop, num)（endpoint=True）。最後は stop そのものにする所まで numpy と同じ */
function linspace(start: number, stop: number, num: number): Float64Array {
  const y = new Float64Array(num);
  const div = num - 1;
  const delta = stop - start;
  if (div > 0) {
    const step = delta / div;
    for (let i = 0; i < num; i++) y[i] = (step === 0 ? (i / div) * delta : i * step) + start;
    y[num - 1] = stop;
  } else {
    for (let i = 0; i < num; i++) y[i] = i * delta + start;
  }
  return y;
}

/** prettypitch/pitch.py の _asym_tukey_window: 頭 lead 割合を 0→1、尻 trail 割合を 1→0、中は 1 */
function asymTukeyWindow(n: number, lead: number, trail: number): Float64Array {
  const w = new Float64Array(n).fill(1);
  if (n <= 1) return w;
  lead = Math.max(0.0, lead);
  trail = Math.max(0.0, trail);
  if (lead + trail > 1.0) {
    const s = lead + trail;
    lead = lead / s;
    trail = trail / s;
  }
  const nl = Math.min(roundHalfEven(lead * n), n);
  const nt = Math.min(roundHalfEven(trail * n), n - nl);
  if (nl > 0) {
    const t = linspace(0.0, Math.PI / 2, nl);
    for (let i = 0; i < nl; i++) w[i] = Math.sin(t[i]) ** 2;
  }
  if (nt > 0) {
    const t = linspace(Math.PI / 2, 0.0, nt);
    for (let i = 0; i < nt; i++) w[n - nt + i] = Math.sin(t[i]) ** 2;
  }
  return w;
}

/** prettypitch/pitch.py の hz_to_midi（1 Hz 未満は 1 Hz） */
const hzToMidi = (hz: number): number => 69.0 + 12.0 * Math.log2(Math.max(hz, 1.0) / 440.0);
/** prettypitch/pitch.py の midi_to_hz */
const midiToHz = (m: number): number => 440.0 * Math.pow(2.0, (m - 69.0) / 12.0);

// ── prettypitch/baseline.py ──

/** fill_values: 値のある音素の値を後ろへ持ち回り、頭は最初の値で埋める。値が 1 つも無ければ全部 NaN のまま */
function fillValues(v: ArrayLike<number>): Float64Array {
  const out = Float64Array.from(v);
  let first = -1;
  for (let i = 0; i < out.length; i++) if (Number.isFinite(out[i])) { first = i; break; }
  if (first < 0) return out;
  let last = NaN;
  for (let i = 0; i < out.length; i++) {
    if (Number.isFinite(out[i])) last = out[i];
    else out[i] = last;
  }
  for (let i = 0; i < first; i++) out[i] = out[first];
  return out;
}

/** phoneme_frame_bounds: 秒の累積（float64、頭から順に足す）× フレームレートを偶数丸め。最後は T に止める */
function phonemeFrameBounds(durSec: ArrayLike<number>, frameRate: number, T: number): Int32Array {
  const fb = new Int32Array(durSec.length + 1);
  let cum = 0.0;
  for (let i = 0; i < durSec.length; i++) {
    cum += durSec[i];
    fb[i + 1] = roundHalfEven(cum * frameRate);
  }
  fb[durSec.length] = T;
  return fb;
}

/** build_baseline: 値付き音素の階段 → フレーム → ガウシアン（MIDI のまま）→ log2 Hz（float32）。値が無ければ null */
function buildBaseline(values: ArrayLike<number>, durSec: ArrayLike<number>, T: number, frameRate: number,
                       sigmaSec: number): Float32Array | null {
  const phVal = fillValues(values);
  for (let i = 0; i < phVal.length; i++) if (!Number.isFinite(phVal[i])) return null;
  const fb = phonemeFrameBounds(durSec, frameRate, T);
  const step = new Float64Array(T); // Python は np.empty。区間は必ず全フレームを覆うので初期値は使われない
  for (let k = 0; k < phVal.length; k++) {
    const a = fb[k];
    const b = fb[k + 1];
    for (let f = a; f < b; f++) step[f] = phVal[k];
  }
  const sm = gaussianFilter1d(step, Math.max(sigmaSec * frameRate, 1e-6));
  const out = new Float32Array(T);
  for (let i = 0; i < T; i++) out[i] = LOG2_440 + (sm[i] - 69.0) / 12.0;
  return out;
}

// ── prepareF0: _f0 のモデルの前まで ──

/**
 * engine/pipeline.py の _f0 のうち、モデルを走らせる前の所。
 * render._f0_score_from_seq → infer._read_score → infer._frame_grid → 土台（核音素の音高だけ・2 つの逃げ道）→
 * fill_values → pitch_midi → style_mix。
 */
/**
 * F0 モデルの伸ばしの記号（engine/pipeline.py の f0_hold）: f0.json の hold_symbol（無ければ「-」）が音素表にあればその記号、
 * 無ければ null（伸ばしの母音は、今までどおり母音を繰り返して渡す）
 */
export function f0Hold(vb: VoicebankWeb): string | null {
  const hs = (vb.f0.hold_symbol as string | undefined) || "-";
  return (vb.f0.phonemes as string[]).includes(hs) ? hs : null;
}

export function prepareF0(vb: VoicebankWeb, seq: PhonemeSequence, spk: number): F0Prep {
  const cfg = vb.f0;
  const frameRate = Number(cfg.frame_rate);
  const ph2id = new Map<string, number>((cfg.phonemes as string[]).map((p, i) => [p, i]));
  const nucleus = new Set<string>(cfg.nucleus as string[]);
  const L = seq.phonemes.length;

  // _f0_score_from_seq: note_midi > 0 の音素にだけ midi を付ける（pau は付かない）。_read_score で検証する。
  // 伸ばしの記号で書いたラベルで学習した F0 モデル（f0Hold）には、伸ばしの母音（seq.held）をその記号で渡す（音高はその音符のまま）
  const hold = f0Hold(vb);
  const phStrs = seq.phonemes.map((p, i) => (hold !== null && seq.held?.[i] ? hold : p));
  const unknown = [...new Set(phStrs.filter((p) => !ph2id.has(p)))].sort();
  if (unknown.length) throw new Error(`phonemes not in the model vocab: ${JSON.stringify(unknown)}`);
  const phIds = new Int32Array(L);
  const durSec = new Float32Array(L); // float(dur) を float32 の配列に入れる（seq.dur_sec がもともと float32）
  const values = new Float64Array(L);
  const nucMissing: string[] = [];
  for (let i = 0; i < L; i++) {
    phIds[i] = ph2id.get(phStrs[i])!;
    durSec[i] = seq.dur_sec[i];
    const midi = Math.trunc(seq.note_midi[i]);
    values[i] = midi > 0 ? midi : NaN;
    if (nucleus.has(phStrs[i]) && !(midi > 0)) nucMissing.push(phStrs[i]);
  }
  if (nucMissing.length) throw new Error(`nucleus phonemes need a midi value: ${JSON.stringify(nucMissing)}`);

  // _frame_grid: T0 = round(np.sum(float32) * frame_rate)。和も積も float32（NumPy 2 では Python の float は弱い型）
  const T0 = roundHalfEven(fr32(npSum(durSec, true) * frameRate));
  const fb = phonemeFrameBounds(durSec, frameRate, T0);
  const phDurs = new Int32Array(L);
  let T = 0;
  for (let i = 0; i < L; i++) {
    phDurs[i] = Math.max(fb[i + 1] - fb[i], 1);
    T += phDurs[i];
  }

  // 土台は学習時と同じく核音素の音高だけから作る（子音は前の核の音高を引き継ぎ、段は核の頭に来る）
  const nucVals = new Float64Array(L);
  for (let i = 0; i < L; i++) nucVals[i] = nucleus.has(phStrs[i]) ? values[i] : NaN;
  const sigma = Number(cfg.baseline_sigma_sec ?? GAUSSIAN_SIGMA_SEC);
  let baseline = buildBaseline(nucVals, durSec, T, frameRate, sigma);
  // 核音素の無い句（子音だけの音符など）: 子音の音高で作る。それも無ければ一定
  if (baseline === null) baseline = buildBaseline(values, durSec, T, frameRate, sigma);
  if (baseline === null) baseline = new Float32Array(T).fill(LOG2_440 + (Number(cfg.pitch_center) - 69.0) / 12.0);

  const filled = fillValues(nucVals);
  const pitchMidi = new Float32Array(L);
  for (let i = 0; i < L; i++) pitchMidi[i] = Number.isFinite(filled[i]) ? filled[i] : Number(cfg.pitch_center);

  const nStyles = Number(cfg.n_styles ?? 1);
  const styleMix = new Float32Array(nStyles);
  styleMix[0] = 1.0;
  return { phStrs, phIds, durSec, values, T, phDurs, baseline, pitchMidi, styleMix, spk, nNoise: Number(cfg.n_noise) };
}

/** _f0 で s_f0.run に渡す入力そのもの。noise は [1, n_noise, 2, T]（AI リテイクの差し替えは呼ぶ側で済ませておく） */
export function f0Feeds(vb: VoicebankWeb, prep: F0Prep, spk: number, noise: Float32Array): Feeds {
  void vb;
  const L = prep.phIds.length;
  const want = prep.nNoise * 2 * prep.T;
  if (noise.length !== want) throw new Error(`F0 の雑音の長さが違う（${noise.length}、要るのは 1×${prep.nNoise}×2×${prep.T}）`);
  return {
    tokens: i64(prep.phIds, [1, L]),
    pitch_midi: f32(prep.pitchMidi, [1, L]),
    durations: i64(prep.phDurs, [1, L]),
    spk_id: i64([spk], [1]),
    style_mix: f32(prep.styleMix, [1, prep.styleMix.length]),
    noise: f32(noise, [1, prep.nNoise, 2, prep.T]),
  };
}

/** _retake で使う、フレームごとの音符の番号: np.repeat(note_index, ph_durs)[:T] */
export function frameNoteIndex(noteIndex: number[], phDurs: Int32Array, T: number): Int32Array {
  let total = 0;
  for (let k = 0; k < phDurs.length; k++) total += Math.max(phDurs[k], 0);
  const out = new Int32Array(Math.min(total, T));
  let f = 0;
  for (let k = 0; k < phDurs.length && f < out.length; k++) {
    for (let j = 0; j < phDurs[k] && f < out.length; j++) out[f++] = noteIndex[k];
  }
  return out;
}

// ── prettypitch/correction.py ──

/** build_per_frame_arrays: フレームごとの音素・目標の音高（休符は 0）・音符の頭 */
function buildPerFrameArrays(phStrs: string[], phDurs: Int32Array, values: Float64Array, T: number,
                             silence: Set<string>) {
  const filled = fillValues(values);
  const phonemesPf: string[] = new Array(T).fill("pau");
  const notePitchesPf = new Float64Array(T);
  const noteBoundariesPf = new Uint8Array(T);
  let cursor = 0;
  let prevSil: boolean | null = null;
  for (let k = 0; k < phStrs.length; k++) {
    const end = Math.min(cursor + phDurs[k], T);
    if (end <= cursor) continue;
    const ph = phStrs[k];
    const isSil = silence.has(ph);
    for (let f = cursor; f < end; f++) phonemesPf[f] = ph;
    if (!isSil && Number.isFinite(filled[k])) for (let f = cursor; f < end; f++) notePitchesPf[f] = filled[k];
    if (cursor === 0 || Number.isFinite(values[k]) || (isSil && prevSil !== true)) noteBoundariesPf[cursor] = 1;
    prevSil = isSil;
    cursor = end;
    if (cursor >= T) break;
  }
  return { phonemesPf, notePitchesPf, noteBoundariesPf };
}

/** 音符の区間（音符の頭から次の頭まで）。np.where(note_boundaries_pf) と np.append(starts[1:], n) */
function noteSegments(bnd: Uint8Array): Array<[number, number]> {
  const starts: number[] = [];
  for (let i = 0; i < bnd.length; i++) if (bnd[i]) starts.push(i);
  return starts.map((s, k) => [s, k + 1 < starts.length ? starts[k + 1] : bnd.length]);
}

/** _segment_correction_gaussian: 重み付き平均の知覚の音高から、音符 1 つぶんの一定の補正量（半音）を出す */
function segmentCorrectionGaussian(f0St: Float64Array, wP: Float64Array, target: number, sgWindow: number,
                                   sigmaSt: number, tukeyAlpha: number): number {
  const n = f0St.length;
  if (n === 0) return 0.0;
  const wE = tukeyWindow(n, tukeyAlpha);
  let win = Math.min(sgWindow, n);
  if (win % 2 === 0) win -= 1;
  const poly = Math.min(SG_POLY_ORDER, win - 1);
  const wD = new Float64Array(n).fill(1);
  if (win >= 3 && poly >= 1) {
    const sm = savgolFilterInterp(f0St, win, poly);
    for (let i = 0; i < n; i++) wD[i] = 1.0 / Math.min(1.0 + 27.0 * Math.abs(f0St[i] - sm[i]), 15.0);
  }
  const w = new Float64Array(n);
  const fw = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const z = (f0St[i] - target) / sigmaSt;
    const wT = Math.exp(-0.5 * (z * z));
    w[i] = wE[i] * wD[i] * wP[i] * wT;
    fw[i] = f0St[i] * w[i];
  }
  const total = npSum(w);
  if (total === 0.0) return 0.0;
  // np.average(f0_st, weights=w) = sum(f0·w) / sum(w)
  const perceived = npSum(fw) / total;
  return target - perceived;
}

/** apply_pitch_correction（第 1 段）: 音符ごとに一定の補正量、休符は 0、つないで σ 30 ms で丸めて足す */
function applyPitchCorrection(f0Hz: Float32Array, voiced: Uint8Array, phonemesPf: string[], notePitchesPf: Float64Array,
                              noteBoundariesPf: Uint8Array, frameRate: number, nucleus: Set<string>,
                              silence: Set<string>, consWeight: number): Float32Array {
  const n = f0Hz.length;
  let sgWindow = Math.max(3, roundHalfEven(SG_WINDOW_MS * 1e-3 * frameRate));
  if (sgWindow % 2 === 0) sgWindow += 1;
  const smoothSigma = SMOOTH_SIGMA_MS * 1e-3 * frameRate;
  const f0St = new Float64Array(n);
  for (let i = 0; i < n; i++) if (voiced[i]) f0St[i] = hzToMidi(f0Hz[i]);
  // _ph_weight: 無音 0・核 1・その他（子音・cl）は cons_weight。無声のフレームは 0
  const wpAll = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = phonemesPf[i];
    const w = silence.has(p) ? 0.0 : nucleus.has(p) ? 1.0 : consWeight;
    wpAll[i] = w * (voiced[i] ? 1.0 : 0.0);
  }
  const correction = new Float64Array(n);
  for (const [s, e] of noteSegments(noteBoundariesPf)) {
    const pitch = notePitchesPf[s];
    if (pitch === 0) continue; // 休符は補正しない
    const c = segmentCorrectionGaussian(f0St.subarray(s, e), wpAll.subarray(s, e), pitch, sgWindow,
                                        TARGET_SIGMA_ST, TUKEY_ALPHA);
    for (let f = s; f < e; f++) correction[f] = c;
  }
  const sm = gaussianFilter1d(correction, smoothSigma);
  const out = Float32Array.from(f0Hz);
  for (let i = 0; i < n; i++) if (voiced[i]) out[i] = midiToHz(f0St[i] + sm[i]);
  return out;
}

/** apply_hann_pitch_correction（第 2 段、window='tukey'）: 核音素のフレームに 窓 × α × (目標 − F0) を当て、σ 11 ms で丸めて足す */
function applyHannPitchCorrection(f0Hz: Float32Array, voiced: Uint8Array, phonemesPf: string[],
                                  notePitchesPf: Float64Array, noteBoundariesPf: Uint8Array, frameRate: number,
                                  alpha: number, nucleus: Set<string>, lead: number, trail: number): Float32Array {
  const n = f0Hz.length;
  const smoothSigma = HANN_SMOOTH_SIGMA_MS * 1e-3 * frameRate;
  const f0St = new Float64Array(n);
  for (let i = 0; i < n; i++) if (voiced[i]) f0St[i] = hzToMidi(f0Hz[i]);
  const correction = new Float64Array(n);
  for (const [s, e] of noteSegments(noteBoundariesPf)) {
    const pitch = notePitchesPf[s];
    if (pitch === 0) continue;
    const idx: number[] = [];
    for (let f = s; f < e; f++) if (nucleus.has(phonemesPf[f]) && voiced[f]) idx.push(f);
    if (idx.length === 0) continue;
    const win = asymTukeyWindow(idx.length, lead, trail);
    for (let j = 0; j < idx.length; j++) correction[idx[j]] = alpha * win[j] * (pitch - f0St[idx[j]]);
  }
  const sm = gaussianFilter1d(correction, smoothSigma);
  const out = Float32Array.from(f0Hz);
  for (let i = 0; i < n; i++) if (voiced[i]) out[i] = midiToHz(f0St[i] + sm[i]);
  return out;
}

/** 試験用（scipy・numpy を写した小道具を、短い列などの端の場合で直接比べるため）。エンジンからは使わない */
export const _internals = { roundHalfEven, npSum, gaussianFilter1d, savgolFilterInterp, tukeyWindow, asymTukeyWindow,
                            buildBaseline, fillValues };

// ── finishF0: _f0 のモデルの後と analyze の変換 ──

/**
 * engine/pipeline.py の _f0 のモデルの後（plain_delta・vib_delta・build_per_frame_arrays・frame_is_pau・plain_hz・uv・
 * correct_f0 を既定のまま）と、analyze の plain_log2 = log2(max(plain_hz, 1e-3))（float32）など。
 * out はモデルの出力（plain・vibrato・vuv、それぞれ [1, T'] を平らにしたもの）
 */
export function finishF0(vb: VoicebankWeb,
                         prep: F0Prep,
                         out: { plain: Float32Array; vibrato: Float32Array; vuv: Float32Array }): F0Result {
  const cfg = vb.f0;
  const frameRate = Number(cfg.frame_rate);
  const nucleus = new Set<string>(cfg.nucleus as string[]);
  // correction.py の _SILENCE_PHS は言語パック（dict/ja.yaml の silence:）から読んだもの。ボイスバンクの音素表の silence と同じ
  const silence = new Set<string>(vb.phoneset.silence);
  const n = Math.min(prep.T, out.plain.length);

  const plainDelta = out.plain.slice(0, n); // float32 → float64 にしても値は同じ
  const vibLog2 = out.vibrato.slice(0, n);
  const baseLog2 = prep.baseline.slice(0, n);
  const { phonemesPf, notePitchesPf, noteBoundariesPf } =
    buildPerFrameArrays(prep.phStrs, prep.phDurs, prep.values, n, silence);
  const frameIsPau = new Uint8Array(n);
  for (let i = 0; i < n; i++) frameIsPau[i] = silence.has(phonemesPf[i]) ? 1 : 0;

  // plain_hz = 2^(baseline + plain_delta) を float64 で計算して float32 に
  const plainHz0 = new Float32Array(n);
  for (let i = 0; i < n; i++) plainHz0[i] = Math.pow(2.0, baseLog2[i] + plainDelta[i]);
  const uv = new Float32Array(n);
  for (let i = 0; i < n; i++) uv[i] = out.vuv[i] > 0.5 && !frameIsPau[i] ? 1 : 0;

  // correct_f0: 補正は plain だけに当てる（有声 = pau でないフレーム）
  const voiced = new Uint8Array(n);
  for (let i = 0; i < n; i++) voiced[i] = frameIsPau[i] ? 0 : 1;
  let plainHz = applyPitchCorrection(plainHz0, voiced, phonemesPf, notePitchesPf, noteBoundariesPf, frameRate,
                                     nucleus, silence, CONS_WEIGHT);
  if (HANN_ALPHA > 0.0) {
    plainHz = applyHannPitchCorrection(plainHz, voiced, phonemesPf, notePitchesPf, noteBoundariesPf, frameRate,
                                       HANN_ALPHA, nucleus, HANN_LEAD_TAPER, HANN_TRAIL_TAPER);
  }
  // f0_hz = plain_hz × np.power(2.0, vib_delta)。vib_delta が float32 なので累乗も float32（NumPy 2 の型の規則）
  const f0Hz = new Float32Array(n);
  for (let i = 0; i < n; i++) f0Hz[i] = plainHz[i] * fr32(Math.pow(2.0, vibLog2[i]));

  // analyze: np.maximum(float32, 1e-3) も log2 も float32 のまま（NumPy 2 では 1e-3 が float32 になる）
  const floor = fr32(1e-3);
  const plainLog2 = new Float32Array(n);
  for (let i = 0; i < n; i++) plainLog2[i] = Math.log2(Math.max(plainHz[i], floor));

  return {
    T: n, plainLog2, vibLog2, uv, baseLog2,
    phonemesPf, notePitchesPf, noteBoundariesPf, frameIsPau, plainDelta,
    plainHzBeforeCorrection: plainHz0, plainHz, f0Hz,
  };
}
