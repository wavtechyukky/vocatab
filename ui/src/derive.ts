// 音符から作るもの: 音素のタイミングとピッチの線、ブレスを入れられる所。エンジンの結果があればそれから、無ければモック
// （音素はかなの規則、ピッチは PrettyPitch の土台と同じ「音符の高さの階段を σ 20 ms でならした線」だけ）。
import { type BreathClipInfo, type BreathSlot, breathSlots } from "./breath";
import { consonantSec, g2p, VOWELS } from "./kana";
import { activeNotes, FPS, type Note, type Project, secPerTick, type Track } from "./model";
import { evalCurve } from "./params";
import { composeF0, type EPhrase, log2ToMidi } from "./engine";

export interface PhoneSeg {
  ph: string;
  start: number; // 秒
  end: number;
  noteId: number;
  idx: number; // その音符の中の音素の番号
  vowel: boolean;
  baseStart: number; // 手で動かす前の頭
  movable: boolean; // 頭を動かせるか（pau の頭 = 句の最後の音素の終わりは動かせない）
  edited: boolean;
  /** 子音の凹み（エンジンが凹みを返すボイスバンクの子音だけ）: 今の深さ dB・広さ（子音の長さに対する割合）、手で動かしたか */
  dip?: { db: number; w: number; edited: boolean };
}

const MIN_PH = 0.012; // 音素の最短（秒）
export const DIP_DB_MIN = -5; // 子音の凹みの深さの限界（名目 dB。画面では深さ 100%。engine/pipeline.py の DIP_DB_MIN と同じ）
const PAU_MIN = 4 / FPS + 1e-4; // pau の後の頭を左へ動かしても、pau はこれより短くしない（engine/pipeline.py の PAU_MIN_FRAMES）

export interface Derived {
  fps: number;
  pitch: Float32Array; // 最後のピッチ（半音、MIDI の連続値）。声の無い所は NaN
  noDev: Float32Array; // ピッチのずれ（pitchDelta）を足す前。ピッチのペンが使う
  phones: PhoneSeg[];
  noteSec: Map<number, { start: number; end: number }>;
  /** ブレスを入れられる所と入れてあるブレス（ブレスの束が無ければ空） */
  breaths: BreathSlot[];
}

/** ブレスの置き方に要るもの: ボイスバンクのブレスの束の一覧と、このトラックの歌い手の id（無ければブレスは出さない） */
export interface BreathContext {
  clips: BreathClipInfo[];
  singer: number;
}


// エンジンが核かどうかを返さないとき（古いエンジン）の代わり: 日本語の核
const VOWEL_LIKE = /^[aiueoAIUEON]$/;

/** エンジンの結果（句の列）から、音素のタイミングとピッチの線を作る。 */
function fromEngine(project: Project, track: Track, phrases: EPhrase[], notes: Note[],
                    noteSec: Map<number, { start: number; end: number }>, breath?: BreathContext): Derived {
  const phones: PhoneSeg[] = [];
  let nFrames = 0;
  for (const ph of phrases) nFrames = Math.max(nFrames, ph.startFrame + ph.nFrames);
  nFrames += Math.ceil(FPS);
  const pitch = new Float32Array(nFrames).fill(NaN);
  const noDev = new Float32Array(nFrames).fill(NaN);
  const byId = new Map(track.notes.map((n) => [n.id, n]));
  // 滑舌（子音の凹みの既定の深さの倍率）: トラック全体の値 ＋ 曲線、子音の真ん中の時刻の値（エンジンの dip_scales と同じ）
  const spt = secPerTick(project.bpm);
  const artAt = (sec: number): number =>
    Math.min(2, Math.max(0, (track.base.articulation ?? 0) + evalCurve(track.params.articulation, sec / spt, 0)));
  for (const ph of phrases) {
    let f = ph.startFrame;
    const count = new Map<number, number>();
    let prev: PhoneSeg | null = null; // 句の中で直前の音素（pau なら null）
    let pauStart: number | null = null; // 直前が pau なら、その pau の頭（秒）
    let heldStart = -Infinity; // 直前の音素が、まとめた母音で終わっていれば、その最後の母音の頭（秒）
    for (const p of ph.phonemes) {
      if (p.ph !== "pau" && p.note !== null) {
        const idx = count.get(p.note) ?? 0;
        count.set(p.note, idx + 1);
        if (p.held && prev) {
          // 音響モデルでまとめた母音: 前の音素を伸ばして 1 つに見せる（skies - → s k ay（長い）z）。頭は動かせない。
          // 番号（idx）は数えたままにして、同じ音符の後ろの音素（z）の手で動かした頭がずれないようにする
          heldStart = f / FPS;
          prev.end = (f + p.frames) / FPS;
          f += p.frames;
          continue;
        }
        const s = f / FPS;
        // 手で動かした頭: エンジンが当てたずれを引いた所が自然な位置（baseStart）。
        // 画面の値がエンジンに届く前（ドラッグ中）でもすぐ追いかけるよう、今の値との差だけ動かして見せる
        const applied = p.shift ?? 0;
        const want = byId.get(p.note)?.phShift?.[idx] ?? 0;
        // 動かせる頭: 前の音素とつながっている頭と、pau の後の頭（句の頭・句の途中の休符の後）。
        // pau の後は、最初に作られた pau の中だけで左へ伸ばせる（pau は PAU_MIN 残す。エンジンの _apply_shifts と同じ）
        // 子音の凹み: エンジンの既定に、手で動かした深さ・広さを重ねる（動かしていない方は既定のまま）。
        // 手で動かしていない子音は、既定の深さに滑舌の倍率を掛ける（−5 dB で止める。エンジンと同じ）
        const ov = byId.get(p.note)?.phDip?.[idx];
        const e = (f + p.frames) / FPS;
        const defDb = p.dip ? (ov ? p.dip.db : Math.max(DIP_DB_MIN, p.dip.db * artAt((s + e) / 2))) : 0;
        const seg: PhoneSeg = { ph: p.ph, start: s, end: e, noteId: p.note, idx,
                                vowel: p.nucleus ?? VOWEL_LIKE.test(p.ph), baseStart: s - applied,
                                movable: !!prev || pauStart !== null, edited: !!want,
                                dip: p.dip ? { db: ov?.db ?? defDb, w: ov?.w ?? p.dip.w, edited: !!ov } : undefined };
        if (want !== applied) {
          if (prev) {
            // エンジンは、まとめた母音の最後の 1 つ（音符の頭から）より前へは動かさない
            const lo = Math.max(prev.start, heldStart) + MIN_PH;
            const t = Math.min(seg.end - MIN_PH, Math.max(lo, seg.baseStart + want));
            seg.start = t;
            prev.end = t;
          } else if (pauStart !== null) {
            // pau の頭は動かないので、pau の頭 ＋ 元の pau の長さ（= baseStart − pauStart）の中に収める
            seg.start = Math.min(seg.end - MIN_PH, Math.max(pauStart + PAU_MIN, seg.baseStart + want));
          }
        }
        phones.push(seg);
        prev = seg;
        pauStart = null;
        heldStart = -Infinity;
      } else {
        prev = null;
        pauStart = f / FPS;
        heldStart = -Infinity;
      }
      f += p.frames;
    }
    const { log2, noDev: nd } = composeF0(ph, track, project);
    for (let i = 0; i < ph.nFrames; i++) {
      const g = ph.startFrame + i;
      if (g < 0 || g >= nFrames || ph.uv[i] < 0.5) continue; // 声の無い所は線を描かない
      pitch[g] = log2ToMidi(log2[i]);
      noDev[g] = log2ToMidi(nd[i]);
    }
  }
  phones.sort((a, b) => a.start - b.start);
  return { fps: FPS, pitch, noDev, phones, noteSec, breaths: slotsOf(notes, phones, noteSec, breath) };
}

const slotsOf = (notes: Note[], phones: PhoneSeg[], noteSec: Map<number, { start: number; end: number }>,
                 breath?: BreathContext): BreathSlot[] =>
  breath && breath.clips.length ? breathSlots(notes, phones, noteSec, breath.clips, breath.singer) : [];

/** breath: ブレスの束と歌い手。エンジンがつながっていて束のあるボイスバンクのときだけ渡す（無ければブレスの所は出ない） */
export function derive(project: Project, track: Track, engine?: EPhrase[], breath?: BreathContext): Derived {
  const spt = secPerTick(project.bpm);
  // トラックの範囲で切った音符（範囲の外の音符は歌わない）
  const notes = activeNotes(track, project.beatsPerBar).sort((a, b) => a.onset - b.onset);
  const noteSec = new Map<number, { start: number; end: number }>();
  for (const n of notes) noteSec.set(n.id, { start: n.onset * spt, end: (n.onset + n.dur) * spt });
  if (engine && engine.length) return fromEngine(project, track, engine, notes, noteSec, breath);

  const phones = phonemeTiming(notes, noteSec);
  const lastEnd = notes.length ? Math.max(...notes.map((n) => noteSec.get(n.id)!.end)) : 0;
  const nFrames = Math.ceil((lastEnd + 2) * FPS);
  // エンジンが無いときのピッチの線は土台だけ（抑揚もビブラートも付けない）。ピッチのずれ（セント）は最後に足す
  const noDev = baselineCurve(notes, noteSec, nFrames);
  const pitch = new Float32Array(nFrames);
  for (let i = 0; i < nFrames; i++) pitch[i] = noDev[i] + evalCurve(track.params.pitchDelta, i / FPS / spt, 0) / 100;
  return { fps: FPS, pitch, noDev, phones, noteSec, breaths: slotsOf(notes, phones, noteSec, breath) };
}

function phonemeTiming(notes: Note[], sec: Map<number, { start: number; end: number }>): PhoneSeg[] {
  const out: PhoneSeg[] = [];
  let prev: Note | null = null;
  for (const n of notes) {
    const { start, end } = sec.get(n.id)!;
    const phs = g2p(n.lyric);
    const last = out[out.length - 1];
    const adjacent = prev !== null && start - sec.get(prev.id)!.end < 0.03;
    if (phs.length === 0) {
      // 伸ばし: 前の母音を続ける
      if (last && adjacent) last.end = end;
      prev = n;
      continue;
    }
    const nucleus = phs[phs.length - 1];
    const cons = phs.slice(0, -1);
    // 子音は音符の頭より前に置く。前の音符の 4 割までに収める
    let total = cons.reduce((s, p) => s + consonantSec(p), 0);
    const room = adjacent ? 0.4 * (sec.get(prev!.id)!.end - sec.get(prev!.id)!.start) : 0.15;
    const scale = total > room ? room / total : 1;
    total *= scale;
    let t = start - total;
    if (last && adjacent) last.end = t; // つながっている音符は隙間なく並べる
    const seg = (ph: string, idx: number, s: number, e: number, vowel: boolean): PhoneSeg =>
      ({ ph, start: s, end: e, noteId: n.id, idx, vowel, baseStart: s, movable: false, edited: false });
    cons.forEach((c, k) => {
      const d = consonantSec(c) * scale;
      out.push(seg(c, k, t, k === cons.length - 1 ? start : t + d, false));
      t += d;
    });
    out.push(seg(nucleus, cons.length, start, end, VOWELS.has(nucleus)));
    prev = n;
  }
  // 手で動かした境界を当てる。前の音素とつながっている頭だけ動かせる（句の頭と終わりは動かない）
  const byId = new Map(notes.map((n) => [n.id, n]));
  for (let i = 0; i < out.length; i++) {
    const s = out[i];
    const p = out[i - 1];
    s.movable = !!p && Math.abs(p.end - s.start) < 1e-6;
    if (!s.movable) continue;
    const sh = byId.get(s.noteId)?.phShift?.[s.idx];
    if (!sh) continue;
    const tt = Math.min(s.end - MIN_PH, Math.max(p.start + MIN_PH, s.baseStart + sh));
    s.start = tt;
    p.end = tt;
    s.edited = true;
  }
  return out;
}

const BASELINE_SIGMA = 0.02; // 秒（PrettyPitch の baseline_sigma_sec と同じ）

/**
 * エンジンが無いときのピッチの線: PrettyPitch の土台と同じく、音符の高さの階段を σ 20 ms のガウシアンでならしたもの。
 * 階段の段は音符の頭（モックの子音は頭より前に置くので、子音の間は前の音符の高さ = 学習時と同じ詰め方）。
 * 休符（音符の無い所）は NaN のまま線を切り、ならすときは値のある所だけで重みを割る（句の端が垂れない）。
 */
function baselineCurve(notes: Note[], sec: Map<number, { start: number; end: number }>, nFrames: number): Float32Array {
  const step = new Float32Array(nFrames).fill(NaN);
  for (const n of notes) {
    const { start, end } = sec.get(n.id)!;
    for (let i = Math.max(0, Math.round(start * FPS)); i < Math.min(nFrames, Math.round(end * FPS)); i++) step[i] = n.pitch;
  }
  const s = BASELINE_SIGMA * FPS;
  const r = Math.ceil(4 * s);
  const ker = Array.from({ length: 2 * r + 1 }, (_, k) => Math.exp(-0.5 * ((k - r) / s) ** 2));
  const out = new Float32Array(nFrames).fill(NaN);
  for (let i = 0; i < nFrames; i++) {
    if (Number.isNaN(step[i])) continue;
    let sum = 0;
    let wsum = 0;
    for (let k = -r; k <= r; k++) {
      const v = step[i + k];
      if (v === undefined || Number.isNaN(v)) continue;
      sum += ker[k + r] * v;
      wsum += ker[k + r];
    }
    out[i] = sum / wsum;
  }
  return out;
}
