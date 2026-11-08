// エンジン（engine/server.py）とのやり取りと、句ごとの最後の F0 の組み立て。
import type { SingerInfo } from "./engine-web/types";
import { getLang } from "./i18n";
import { FPS, type Project, secPerTick, type Track } from "./model";
import { evalCurve, MULTIPLY_KEYS, paramDef, VOICE_KEYS } from "./params";

export interface EngineInfo {
  voicebanks: Array<{ key: string; name: string; language: string; singers: SingerInfo[]; default_vocoder: string }>;
  vocoders: Array<{ key: string; name: string; kind: string; voice_params: boolean }>;
}

/** 歌い手の表示名: 画面が日本語なら日本語のフルネーム、それ以外は英語。ボイスバンクに無ければ内部の名前のまま */
export function singerDisplay(s: SingerInfo | undefined, fallback: string): string {
  const d = s?.display;
  if (!d) return fallback;
  return (getLang() === "ja" ? d.ja : d.en) ?? d.en ?? d.ja ?? fallback;
}

export interface EPhoneme {
  ph: string;
  frames: number;
  note: number | null; // 画面の音符の id（pau は null）
  shift?: number; // エンジンが当てた、手で動かした頭のずれ（秒）
  nucleus?: boolean; // 音節の核（母音など。言語パックの nucleus）か。古いエンジンには無い
  /** 子音の既定の凹み（深さ dB・広さ = 子音の長さに対する割合。谷は子音の頭から）。凹みを掛けないボイスバンク・母音には無い */
  dip?: { db: number; w: number };
}

export type PhraseStatus = "解析済み" | "合成中" | "済み" | "失敗";

/** エンジンが返した 1 句。startFrame は曲の頭からの絶対フレーム（172.27 fps）。 */
export interface EPhrase {
  key: string;
  noteIds: number[];
  startFrame: number;
  startSec: number; // 句の頭の絶対時刻（丸める前。待機から起きたエンジンへ解析の結果を戻すときに要る）
  nFrames: number;
  phonemes: EPhoneme[];
  plain: Float32Array; // 補正後の抑揚の線（log2 Hz）
  base: Float32Array; // 抑揚の土台（音高の階段を σ 20 ms で丸めたもの、log2 Hz）。抑揚 = plain − base
  vib: Float32Array; // ビブラートの成分（log2）
  uv: Float32Array; // 有声 = 1
  seed: number; // 乱数の種（解析のときの AnalyzeReq.seed。戻すときに要る）
  status: PhraseStatus;
  f0Hash?: string; // 合成したときの F0 の指紋（同じなら合成し直さない）
  wav?: Float32Array;
  melIn?: Float32Array; // ① ボコーダの入力（bin ごと [128 × nFrames]）
  melOut?: Float32Array; // ③ 出た音の mel
  melPlain?: Float32Array; // ② ボイスパラメータを掛けない音の mel（② の行がオンのときだけ）
}

/**
 * 句のボイスパラメータ（フレームごと）。トラック全体の値と曲線を合わせる（有声度は掛け算、ほかは足し算）。
 * どのフレームも既定値のものは入れない（エンジンは無いものを既定値として扱う）。
 */
export function phraseParams(ph: EPhrase, track: Track, project: Project): Record<string, Float32Array> {
  const spt = secPerTick(project.bpm);
  const out: Record<string, Float32Array> = {};
  for (const key of VOICE_KEYS) {
    const d = paramDef(key);
    const base = track.base[key as keyof typeof track.base];
    const curve = track.params[key];
    if (!curve && base === d.def) continue;
    const a = new Float32Array(ph.nFrames);
    let neutral = true;
    for (let i = 0; i < ph.nFrames; i++) {
      const cv = evalCurve(curve, (ph.startFrame + i) / FPS / spt, d.def);
      const v = Math.min(d.max, Math.max(d.min, MULTIPLY_KEYS.has(key) ? base * cv : base + cv));
      a[i] = v;
      if (v !== d.def) neutral = false;
    }
    if (!neutral) out[key] = a;
  }
  return out;
}

export function unb64(s: string): Float32Array {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

export function b64(a: Float32Array): string {
  const u8 = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  return btoa(s);
}

/** 配列の指紋（FNV-1a）。合成し直すかどうかの判定に使う。 */
export function hashF32(a: Float32Array, extra: string): string {
  const u = new Uint32Array(a.buffer, a.byteOffset, a.length);
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) {
    h ^= u[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return `${h.toString(16)}:${a.length}:${extra}`;
}

/** 句のフレームごとに、どの音符に属するか（音素の note から）。 */
export function frameNotes(ph: EPhrase): Int32Array {
  const fn = new Int32Array(ph.nFrames).fill(-1);
  let f = 0;
  for (const p of ph.phonemes) {
    for (let k = 0; k < p.frames && f < ph.nFrames; k++, f++) fn[f] = p.note ?? -1;
  }
  return fn;
}

const IMOD_FADE_SEC = 0.05; // 音符ごとの抑揚の倍率を、音符の境目の前後この秒数でなだらかにつなぐ

/**
 * 音符ごとの抑揚の倍率をフレームに並べ、境目をなだらかにする。
 * 抑揚（土台からのずれ）は音符の境目の遷移で大きいので、倍率が境目で段になると F0 が跳ぶ。pau は隣の音符の値で埋める。
 */
function intonationMods(ph: EPhrase, notes: Map<number, { attrs: { iMod?: number } }>, fn: Int32Array): Float32Array {
  const n = ph.nFrames;
  const raw = new Float32Array(n).fill(NaN);
  let any = false;
  for (let i = 0; i < n; i++) {
    if (fn[i] < 0) continue;
    raw[i] = notes.get(fn[i])?.attrs.iMod ?? 1;
    if (raw[i] !== 1) any = true;
  }
  if (!any) return raw.fill(1);
  let last = NaN;
  for (let i = 0; i < n; i++) (Number.isNaN(raw[i]) ? (raw[i] = last) : (last = raw[i]));
  last = NaN;
  for (let i = n - 1; i >= 0; i--) (Number.isNaN(raw[i]) ? (raw[i] = last) : (last = raw[i]));
  if (Number.isNaN(raw[0])) return raw.fill(1);
  // 移動平均（端は端の値を延ばす）
  const k = Math.max(1, Math.round(IMOD_FADE_SEC * FPS));
  const cum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) cum[i + 1] = cum[i] + raw[i];
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = i - k;
    const b = i + k;
    const s = cum[Math.min(n, b + 1)] - cum[Math.max(0, a)] + Math.max(0, -a) * raw[0] + Math.max(0, b - n + 1) * raw[n - 1];
    out[i] = s / (2 * k + 1);
  }
  return out;
}

/**
 * 句の最後の F0（log2 Hz）を組み立てる。無声の所も途切れずに作る（ボコーダと音響モデルに渡すため）。
 *   F0 = 土台 + 抑揚 × 抑揚の倍率（音符ごと）× 抑揚の強さ（曲線）
 *        + ビブラートの成分 × ビブラートの変調（音符ごと）× ビブラートの強さ（曲線）+ ピッチのずれ
 * 土台は音高の階段を丸めたもの、抑揚は補正後の抑揚の線と土台の差（PrettyPitch の baseline と plain_delta）。
 * ビブラートの変調は今は 0〜2 の倍率として掛ける（1 を超えた分を「幅が一定のビブラート」にするのは後で）。
 */
export function composeF0(ph: EPhrase, track: Track, project: Project): { log2: Float32Array; noDev: Float32Array } {
  const spt = secPerTick(project.bpm);
  const notes = new Map(track.notes.map((n) => [n.id, n]));
  const fn = frameNotes(ph);
  const imod = intonationMods(ph, notes, fn);
  const envDef = paramDef("vibratoEnv").def;
  const intDef = paramDef("intonationEnv").def;
  const log2 = new Float32Array(ph.nFrames);
  const noDev = new Float32Array(ph.nFrames);
  for (let i = 0; i < ph.nFrames; i++) {
    const tick = (ph.startFrame + i) / FPS / spt;
    const mod = notes.get(fn[i])?.attrs.vMod ?? 1;
    const env = evalCurve(track.params.vibratoEnv, tick, envDef);
    const ie = imod[i] * evalCurve(track.params.intonationEnv, tick, intDef);
    const v = ph.base[i] + (ph.plain[i] - ph.base[i]) * ie + ph.vib[i] * mod * env;
    noDev[i] = v;
    log2[i] = v + evalCurve(track.params.pitchDelta, tick, 0) / 1200;
  }
  return { log2, noDev };
}

export const log2ToMidi = (l: number): number => 69 + 12 * (l - Math.log2(440));
