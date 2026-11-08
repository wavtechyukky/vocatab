// プロジェクトの形。時間は拍の刻み（4 分音符 = TPQ）で持ち、秒への変換はテンポから行う。

export const TPQ = 480;
export const FPS = 44100 / 256; // 音響モデル・ボコーダの格子（172.27 フレーム/秒）

export interface NoteAttrs {
  vMod: number; // ビブラートの変調（0〜2。AI のビブラートの成分に掛ける倍率）
  iMod?: number; // 抑揚の倍率（0〜2。AI の抑揚 = 音高の階段からのずれに掛ける。無ければ 1）
  // AI リテイクの種（0 = 元のテイク）。その音符の所だけ、モデルの乱数を差し替える
  retakeF0?: number; // ピッチ（PrettyPitch の拡散の雑音）
  retakeMel?: number; // 声（LeapSinger の励振の雑音）
  retakeF0Prev?: number[]; // それまでのテイクの種（「一つ前」で戻る。古い順）
  retakeMelPrev?: number[];
}

export interface Note {
  id: number;
  onset: number; // 刻み
  dur: number; // 刻み
  pitch: number; // MIDI の番号
  lyric: string;
  attrs: NoteAttrs;
  /** 手で動かした音素の頭（この音符の中の音素の番号 → 既定の位置からのずれ、秒）。句の頭は動かせない */
  phShift?: Record<number, number>;
  /** 手で動かした子音の凹み（この音符の中の音素の番号 → 深さ dB・広さ。広さは子音の長さに対する割合で、谷は子音の頭から）。
   *  無い項目はボイスバンクの既定のまま */
  phDip?: Record<number, { db?: number; w?: number }>;
  /** この音符の前に入れたブレス（ボイスバンクのブレスの束の中の番号）。無ければ入れていない（breath.ts） */
  breath?: { index: number };
}

/** トラック全体の声の設定（フレームごとのパラメータの曲線はこれに足す）。 */
export interface VoiceBase {
  loudness: number; // dB
  tension: number; // −1〜1
  breathiness: number; // −1〜1
  gender: number; // −1〜1
  voicing: number; // 0〜1
  consonant: number; // 子音の強さ −1〜1
  resonance: number; // 響き −1〜1
  growl: number; // うなり 0〜1
  articulation: number; // 滑舌: 子音の凹みの既定の深さの倍率 0〜2（既定 0 = 凹み無し、1 = レシピどおり。手で動かした凹みには掛けない）
}

export type ParamKey =
  "pitchDelta" | "intonationEnv" | "vibratoEnv" | "loudness" | "tension" | "breathiness" | "gender" | "voicing" | "consonant" |
  "resonance" | "growl" | "articulation";
export type Interp = "linear" | "cosine" | "cubic";

/** フレームごとのパラメータの曲線。点は [刻み, 値] を刻みの順に並べる。点の外側は既定値。 */
export interface Curve {
  pts: Array<[number, number]>;
  interp: Interp;
}

export interface Track {
  id: number;
  name: string;
  voice: string;
  /** 歌わせるボイスバンク（エンジンの voicebanks の key）。無ければ一覧の先頭 */
  voicebank?: string;
  /** トラックの色（"#rrggbb"）。空なら差し色から作る自動の色（theme.ts の trackColor） */
  color: string;
  notes: Note[];
  mute: boolean;
  solo: boolean;
  gain?: number; // トラックの音量（dB、既定 0）。再生のときに掛ける（合成し直さない）
  pan?: number; // −1（左）〜 1（右）、既定 0
  /** トラックの範囲（刻み）。この外の音符は無いもの、端をまたぐ音符は端で切れたものとして歌わせる。無ければ音符から決める */
  region?: Region;
  /** 歌のトラック（既定）か、伴奏などの音声を鳴らすトラックか */
  kind?: "vocal" | "audio";
  audio?: AudioRef;
  /** エフェクト（再生と書き出しで掛ける。合成し直さない）。無ければ全部切 */
  fx?: TrackFx;
  base: VoiceBase;
  params: Partial<Record<ParamKey, Curve>>;
}

export interface Region {
  start: number;
  end: number;
}

/** 伴奏のトラックの音声（中身は audiofiles.ts に置き、ここには参照だけ）。 */
export interface AudioRef {
  id: string;
  name: string; // ファイルの名前（拡張子なし）
  fileName?: string; // 元のファイル名（拡張子つき。開くときに探す手がかり）
  hash?: string; // 中身の SHA-256（開くときに同じ音声かを確かめる）
  durationSec: number;
  offsetSec: number; // 範囲の頭で鳴らす、音声の中の位置（左の端を詰めると増える）
}

export const isAudioTrack = (t: Track): boolean => t.kind === "audio";

/** イコライザーの帯（山・谷の形。gain は dB）。 */
export interface EqBand {
  freq: number;
  gain: number;
  q: number;
}

export type ReverbPreset = "room" | "hall" | "plate" | "cathedral";

export interface TrackFx {
  eq: { on: boolean; lowCut: number; bands: EqBand[] }; // lowCut: 低域を切る周波数（Hz、12 dB/oct）
  comp: { on: boolean; threshold: number; attack: number; ratio: number }; // dB・秒・N:1
  reverb: { on: boolean; preset: ReverbPreset; preDelay: number; decay: number; wet: number }; // ms・秒・0〜1
}

export function defaultFx(): TrackFx {
  return {
    eq: { on: false, lowCut: 40, bands: [] }, // 山・谷の点はグラフのダブルクリックで足す（最大 fx.ts の EQ_MAX_BANDS）
    comp: { on: false, threshold: -18, attack: 0.01, ratio: 3 },
    reverb: { on: false, preset: "hall", preDelay: 25, decay: 2.4, wet: 0.22 }, // ホールの値（fx.ts の REVERB_PRESETS）
  };
}

/** トラックの範囲。決めていなければ、音符を含む小節の頭から最後の音符の終わりを含む小節の終わりまで。 */
export function trackRegion(t: Track, beatsPerBar: number): Region {
  if (t.region) return t.region;
  const bar = beatsPerBar * TPQ;
  if (!t.notes.length) return { start: 0, end: bar };
  return { start: Math.floor(Math.min(...t.notes.map((n) => n.onset)) / bar) * bar,
           end: Math.ceil(Math.max(...t.notes.map((n) => n.onset + n.dur)) / bar) * bar };
}

/** 新しく描いた音符が範囲の外に出たら、範囲を小節の区切りまで広げる。 */
export function growRegion(t: Track, n: Note, beatsPerBar: number): void {
  const bar = beatsPerBar * TPQ;
  const r = trackRegion(t, beatsPerBar);
  t.region = { start: Math.min(r.start, Math.floor(n.onset / bar) * bar),
               end: Math.max(r.end, Math.ceil((n.onset + n.dur) / bar) * bar) };
}

/** 歌わせる音符: 範囲の外の音符を除き、端をまたぐ音符は端で切った写しにする（id はそのまま）。 */
export function activeNotes(t: Track, beatsPerBar: number): Note[] {
  const r = trackRegion(t, beatsPerBar);
  const out: Note[] = [];
  for (const n of t.notes) {
    const a = Math.max(n.onset, r.start);
    const b = Math.min(n.onset + n.dur, r.end);
    if (b <= a) continue;
    out.push(a === n.onset && b === n.onset + n.dur ? n : { ...n, onset: a, dur: b - a });
  }
  return out;
}

export function defaultBase(): VoiceBase {
  return { loudness: 0, tension: 0, breathiness: 0, gender: 0, voicing: 1, consonant: 0, resonance: 0,
           growl: 0, articulation: 0 };
}

export interface Project {
  bpm: number;
  beatsPerBar: number;
  tracks: Track[];
  activeTrack: number;
}

export function defaultAttrs(): NoteAttrs {
  return { vMod: 1 };
}

/** 音符の設定として持つ項目（読み込むときに、これ以外の古い項目は落とす。projectfile.ts の migrate） */
export const NOTE_ATTR_KEYS: Array<keyof NoteAttrs> = ["vMod", "iMod", "retakeF0", "retakeMel", "retakeF0Prev", "retakeMelPrev"];

export const secPerTick = (bpm: number): number => 60 / (bpm * TPQ);

const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export function pitchName(p: number): string {
  return NAMES[((p % 12) + 12) % 12] + (Math.floor(p / 12) - 1);
}
export const isBlackKey = (p: number): boolean => [1, 3, 6, 8, 10].includes(((p % 12) + 12) % 12);
