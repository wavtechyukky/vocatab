// 句 → 音素ごとの長さ（engine/pipeline.py の PrettyPitchLeapSinger.analyze の ①② と _apply_shifts、
// PrettyPitch の svs/duration_postprocess.py・svs/cons_dur.py の _flatten・duration_model/features.py・
// svs/render.py の phrase_start_sec・_align_durs と同じ）。
//
// 子音の長さのモデル（cons_dur.onnx）は別の所で動かすので、2 段に分けてある:
//   prepareDurations  音節に分け、モデルへの入力（feeds）を作る
//   finishDurations   モデルの出力（dur_sec）を自然長として当て、floor と圧縮・手で動かした分・フレームへの丸めまで
//
// Python は numpy の float32 の配列（PhonemeSequence.dur_sec など）を使う所があるので、そこは Math.fround で
// 同じ丸めにする（np.cumsum も float32 のまま足していく）。ほかは float64（JS の number）のまま。
import { PhoneSet } from "./phoneset";
import type { Feeds, PNote, PhonemeSequence, Phrase, ProfileTable, VoicebankWeb } from "./types";
import { f32, i64 } from "./types";

const PAU = "pau";
/** 子音の長さのモデルが無い・その子音の予測が欠けているときの 1 子音の長さ [s]（FALLBACK_CONS_DUR） */
export const FALLBACK_CONS_DUR = 0.08;
const NUC_EPS = 0.005; // 母音の長さの下限（圧縮しても 0 にしない）
const PAU_DROP_SEC = 0.012; // 音符の間の隙間の残りがこれ未満なら pau を置かず、直前の核へ返す
const DEFAULT_PHONEME_WEIGHT = 1.0;
const DEFAULT_CONS_FLOOR = 0.05;
/** 手で動かしても、音素はこれより短くしない（pipeline.MIN_PH_SEC） */
export const MIN_PH_SEC = 0.012;
/** pau の後の音素の頭を左へ動かしても、pau はこのフレーム数より短くしない（pipeline.PAU_MIN_FRAMES） */
export const PAU_MIN_FRAMES = 4;
/** フレームへの丸めの float32 の誤差で 3 フレームにならないための足し [s]（pipeline.PAU_MIN_EPS） */
export const PAU_MIN_EPS = 1e-4;

const has = (o: object, k: string): boolean => Object.prototype.hasOwnProperty.call(o, k);

const psCache = new WeakMap<VoicebankWeb, PhoneSet>();
function phoneSet(vb: VoicebankWeb): PhoneSet {
  let ps = psCache.get(vb);
  if (!ps) {
    ps = new PhoneSet(vb.phoneset);
    psCache.set(vb, ps);
  }
  return ps;
}

// ── 数の扱いを Python・numpy に合わせる道具 ──

/** Python の sum()（3.10: 左から順に float64 で足す） */
function pySum(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s;
}

/**
 * numpy の sum（pairwise_sum と同じ足し順）。f32 なら途中も float32 で丸める。
 * 8 個未満は左から順、128 個までは 8 本に分けて足してから合わせ、それより多いと半分に割る。
 */
function npSum(a: ArrayLike<number>, lo: number, hi: number, f32round: boolean): number {
  const r = f32round ? Math.fround : (x: number) => x;
  const n = hi - lo;
  if (n < 8) {
    let s = r(0);
    for (let i = lo; i < hi; i++) s = r(s + a[i]);
    return s;
  }
  if (n <= 128) {
    const acc = [0, 0, 0, 0, 0, 0, 0, 0].map((_, j) => r(a[lo + j]));
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) acc[j] = r(acc[j] + a[lo + i + j]);
    let s = r(r(r(acc[0] + acc[1]) + r(acc[2] + acc[3])) + r(r(acc[4] + acc[5]) + r(acc[6] + acc[7])));
    for (; i < n; i++) s = r(s + a[lo + i]);
    return s;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return r(npSum(a, lo, lo + n2, f32round) + npSum(a, lo + n2, hi, f32round));
}

/** np.round・Python の round と同じ（ちょうど半分は偶数へ） */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// ── 子音の長さの下限など（duration_model/profile.py の ConsDurProfile を歌い手ごとに引いた表） ──

export class Profile {
  readonly enabled: boolean;
  private readonly t: ProfileTable;

  constructor(vb: VoicebankWeb, spk: number | null) {
    const p = vb.profile;
    this.enabled = p.enabled !== false;
    const key = spk === null ? "*" : String(spk);
    // 書き出しに無い歌い手は、話者別の上書きが無いのと同じく全話者共通（"*"）
    this.t = (has(p, key) ? p[key] : p["*"]) as ProfileTable;
  }

  vMin(): number {
    return this.t.v_min;
  }

  cMin(c: string): number {
    return has(this.t.c_min, c) ? this.t.c_min[c] : DEFAULT_CONS_FLOOR;
  }

  leadCMin(c: string): number {
    return has(this.t.lead_c_min, c) ? this.t.lead_c_min[c] : this.cMin(c);
  }

  weight(ph: string): number {
    const w = has(this.t.weight, ph) ? this.t.weight[ph] : DEFAULT_PHONEME_WEIGHT;
    return Math.min(20.0, Math.max(0.05, w));
  }
}

// ── 音節に分ける（duration_postprocess.split_syllable・_syllabify・split_phrase_syllables） ──

const isEdgeConsonant = (ps: PhoneSet, ph: string): boolean => !ps.isPau(ph) && !ps.isNucleus(ph);
const isCore = (ps: PhoneSet, ph: string): boolean => ps.isNucleus(ph) || ps.isPau(ph);

/**
 * split_syllable: 音素の列を (onset, nucleus, coda) に分ける（音素は正規化する）。
 * 核が 1 つも無いとき（子音だけ・cl だけ）は ([], 全部, []) にして、音素を音符の区間の中に残す。
 */
export function splitSyllable(phonemes: string[], ps: PhoneSet): [string[], string[], string[]] {
  const phs = phonemes.map((x) => (ps.isPau(x) ? ps.pau : ps.normalize(x)));
  if (phs.length && !phs.some((x) => isCore(ps, x))) return [[], phs, []];
  const n = phs.length;
  let i = 0;
  while (i < n && isEdgeConsonant(ps, phs[i])) i++;
  let j = n;
  while (j > i && isEdgeConsonant(ps, phs[j - 1])) j--;
  return [phs.slice(0, i), phs.slice(i, j), phs.slice(j)];
}

/** _syllabify: 核を最大 1 つ持つまとまりに分ける（核の前に続く子音はその核の側へ） */
export function syllabify(phonemes: string[], ps: PhoneSet): string[][] {
  const phs = phonemes.slice();
  const core: number[] = [];
  phs.forEach((p, i) => { if (isCore(ps, p)) core.push(i); });
  if (core.length <= 1) return [phs];
  const starts = [0];
  for (let q = 0; q + 1 < core.length; q++) {
    const a = core[q];
    let j = core[q + 1];
    while (j > a + 1 && !isCore(ps, phs[j - 1])) j--;
    starts.push(j);
  }
  const ends = starts.slice(1).concat([phs.length]);
  return starts.map((x, q) => phs.slice(x, ends[q]));
}

/** split_phrase_syllables: 核が複数ある音符を、まとまりの数で等分した音符に分ける（隙間は最後のまとまりだけ） */
export function splitPhraseSyllables(phrase: Phrase, ps: PhoneSet): Phrase {
  const notes: PNote[] = [];
  for (const note of phrase.notes) {
    const groups = syllabify(note.phonemes, ps);
    const m = groups.length;
    groups.forEach((g, gi) => {
      notes.push({ phonemes: g.slice(), pitch_midi: note.pitch_midi, dur_sec: note.dur_sec / m,
                   gap_after_sec: gi === m - 1 ? note.gap_after_sec : 0.0, melisma: !!note.melisma && gi === 0 });
    });
  }
  return { notes, leading_pau_sec: phrase.leading_pau_sec, trailing_pau_sec: phrase.trailing_pau_sec,
           start_sec: phrase.start_sec };
}

// ── 子音の長さのモデルへの入力（cons_dur._flatten・features.extract_phrase_records・records_to_batch） ──

type Slot = "onset" | "coda" | "core";

/** cons_dur._flatten: 句 → 音素の列・仮の長さ（音節の総時間を核に置き、子音は 0）・音高・子音の戻し先 */
function flatten(phrase: Phrase, ps: PhoneSet) {
  const phs: string[] = [PAU];
  const durs: number[] = [phrase.leading_pau_sec];
  const vals: number[] = [NaN];
  const where = new Map<number, [number, Slot, number]>();
  phrase.notes.forEach((note, j) => {
    const [onset, nucleus, coda] = splitSyllable(note.phonemes, ps);
    const span = note.dur_sec;
    const per = nucleus.length ? span / nucleus.length : span;
    onset.forEach((c, k) => {
      where.set(phs.length, [j, "onset", k]);
      phs.push(c); durs.push(0.0); vals.push(NaN);
    });
    const coreOnly = nucleus.length > 0 && !nucleus.some((v) => isCore(ps, v));
    if (coreOnly) {
      nucleus.forEach((c, k) => {
        where.set(phs.length, [j, "core", k]);
        phs.push(c); durs.push(k === 0 ? span : 0.0); vals.push(NaN);
      });
    } else {
      for (const v of nucleus) {
        phs.push(v); durs.push(per); vals.push(v === PAU ? NaN : note.pitch_midi);
      }
    }
    if (!nucleus.length) { // 核の無い音符は尺だけ確保
      phs.push(PAU); durs.push(span); vals.push(NaN);
    }
    coda.forEach((c, k) => {
      where.set(phs.length, [j, "coda", k]);
      phs.push(c); durs.push(0.0); vals.push(NaN);
    });
    if (note.gap_after_sec > 0) {
      phs.push(PAU); durs.push(note.gap_after_sec); vals.push(NaN);
    }
  });
  phs.push(PAU); durs.push(phrase.trailing_pau_sec); vals.push(NaN);
  return { phs, durs, vals, where };
}

/** features.cluster_lengths: 音素ごとに、自分が属する子音の連なりの本数（子音でなければ 0） */
function clusterLengths(phs: string[], ps: PhoneSet): number[] {
  const out = new Array<number>(phs.length).fill(0);
  let i = 0;
  while (i < phs.length) {
    if (ps.isPau(phs[i])) { i++; continue; }
    let j = i;
    while (j < phs.length && !ps.isPau(phs[j])) j++;
    let pos = i;
    for (const syl of syllabify(phs.slice(i, j), ps)) {
      const nuc = syl.findIndex((p) => ps.isNucleus(p));
      if (nuc < 0) {
        for (let q = pos; q < pos + syl.length; q++) out[q] = syl.length;
      } else {
        for (let q = pos; q < pos + nuc; q++) out[q] = nuc;
        for (let q = pos + nuc + 1; q < pos + syl.length; q++) out[q] = syl.length - nuc - 1;
      }
      pos += syl.length;
    }
    i = j;
  }
  return out;
}

/** features.syllable_span_durs: 音節の総時間を核に集め、子音は 0 にした配列 */
function syllableSpanDurs(phs: string[], dur: number[], ps: PhoneSet): number[] {
  const out = new Array<number>(phs.length).fill(0);
  let i = 0;
  while (i < phs.length) {
    if (ps.isPau(phs[i])) { out[i] = dur[i]; i++; continue; }
    let j = i;
    while (j < phs.length && !ps.isPau(phs[j])) j++;
    let pos = i;
    for (const syl of syllabify(phs.slice(i, j), ps)) {
      const span = npSum(dur, pos, pos + syl.length, false);
      const nuc = syl.findIndex((p) => ps.isNucleus(p));
      out[nuc >= 0 ? pos + nuc : pos] = span;
      pos += syl.length;
    }
    i = j;
  }
  return out;
}

interface ConsRecord {
  idx: number;
  cons: string;
  prev_ph: string;
  next_ph: string;
  prev2_ph: string;
  next2_ph: string;
  prev_note_dur: number;
  next_note_dur: number;
  pitch_midi: number;
  pitch_diff: number;
  prev_note_exists: number;
  cluster_len: number;
  spk_id: number;
  style_id: number;
}

/** features.extract_phrase_records: 子音ごとの特徴（入るべき音符の無い末尾の子音は作らない。
 *  ただし言語パックが final_consonants のときは、前の音符の終声として作る） */
function extractPhraseRecords(phs: string[], dur: number[], values: number[], spk: number, style: number,
                              ps: PhoneSet): ConsRecord[] {
  const span = syllableSpanDurs(phs, dur, ps);
  const clu = clusterLengths(phs, ps);
  const recs: ConsRecord[] = [];
  const n = phs.length;
  for (let i = 0; i < n; i++) {
    const ph = phs[i];
    if (!ps.consSet.has(ph)) continue;
    let j = -1; // 直前の核（無音で打ち切り）
    for (let jj = i - 1; jj >= 0; jj--) {
      if (ps.isPau(phs[jj])) break;
      if (ps.isNucleus(phs[jj])) { j = jj; break; }
    }
    let prevNoteDur = 0.0;
    let prevVal = NaN;
    if (j >= 0) {
      prevNoteDur = npSum(span, j, i + 1, false);
      prevVal = values[j];
    }
    let k = -1; // 直後の核
    for (let kk = i + 1; kk < n; kk++) {
      if (ps.isPau(phs[kk])) break;
      if (ps.isNucleus(phs[kk])) { k = kk; break; }
    }
    let nextNoteDur: number;
    let pit: number;
    if (k < 0) {
      // 後ろに核の無い子音（句末や休みの前）。既定では入るべき音符が無いので扱わない。
      // final_consonants のとき（終声で句が終わる韓国語など）は前の音符の終声として扱う: 音高は前の音符の代表値、
      // next_note_dur = 0（＝後ろに音符が無い。prev_note_dur = 0 が「前に音符が無い」を表すのと同じ約束）
      if (!(ps.finalConsonants && j >= 0)) continue;
      nextNoteDur = 0.0;
      pit = Number.isFinite(prevVal) ? prevVal : 60.0;
      prevVal = pit;
    } else {
      nextNoteDur = span[k];
      pit = values[k];
      if (!Number.isFinite(pit)) pit = Number.isFinite(prevVal) ? prevVal : 60.0;
      if (!Number.isFinite(prevVal)) prevVal = pit;
    }
    recs.push({
      idx: i, cons: ph,
      prev_ph: i > 0 ? phs[i - 1] : PAU, next_ph: i + 1 < n ? phs[i + 1] : PAU,
      prev2_ph: i > 1 ? phs[i - 2] : PAU, next2_ph: i + 2 < n ? phs[i + 2] : PAU,
      prev_note_dur: prevNoteDur, next_note_dur: nextNoteDur,
      pitch_midi: roundHalfEven(Math.min(Math.max(pit, 0.0), 127.0)),
      pitch_diff: pit - prevVal,
      prev_note_exists: j >= 0 ? 1 : 0,
      cluster_len: clu[i], spk_id: spk, style_id: style,
    });
  }
  return recs;
}

// features._BATCH_KEYS_LONG（int64 で渡す入力）。ほかは float32
const BATCH_KEYS_LONG = new Set(["cons_id", "prev_ph_id", "next_ph_id", "prev2_ph_id", "next2_ph_id",
  "pitch_midi", "prev_note_exists", "cluster_len", "spk_id", "style_id"]);

/** features.records_to_batch のうち、モデルの入力（cons_dur.json の inputs の順）だけを型付きの配列にする */
function recordsToFeeds(recs: ConsRecord[], inputs: string[], ps: PhoneSet): Feeds {
  const src: Record<string, number[]> = {
    cons_id: recs.map((r) => ps.toId(r.cons)),
    prev_ph_id: recs.map((r) => ps.toId(r.prev_ph)),
    next_ph_id: recs.map((r) => ps.toId(r.next_ph)),
    prev2_ph_id: recs.map((r) => ps.toId(r.prev2_ph)),
    next2_ph_id: recs.map((r) => ps.toId(r.next2_ph)),
    pitch_midi: recs.map((r) => r.pitch_midi),
    prev_note_exists: recs.map((r) => r.prev_note_exists),
    cluster_len: recs.map((r) => r.cluster_len),
    spk_id: recs.map((r) => r.spk_id),
    style_id: recs.map((r) => r.style_id),
    prev_note_dur: recs.map((r) => r.prev_note_dur),
    next_note_dur: recs.map((r) => r.next_note_dur),
    pitch_diff: recs.map((r) => r.pitch_diff),
  };
  const feeds: Feeds = {};
  for (const name of inputs) {
    const v = src[name];
    if (!v) throw new Error(`子音の長さのモデルの入力 ${name} を作れません`);
    feeds[name] = BATCH_KEYS_LONG.has(name) ? i64(v, [v.length]) : f32(Float32Array.from(v), [v.length]);
  }
  return feeds;
}

// ── ① の前半: 音節に分け、モデルへの入力を作る ──

export interface DurPrep {
  /** 元の句（音符の番号の対応に使う） */
  phrase: Phrase;
  /** 音節ごとの音符に分けた句（split_phrase_syllables） */
  ph2: Phrase;
  /** 子音の長さのモデル（cons_dur.onnx）への入力。子音のレコードが無ければ null（モデルを呼ばない） */
  feeds: Feeds | null;
  /** レコードごとの戻し先（ph2 の音符の番号・onset/coda/core・その中の番号）。モデルの出力の順と同じ */
  where: Array<[number, Slot, number]>;
}

/** PrettyPitchLeapSinger.analyze の split_phrase_syllables と、_cons_naturals のうちモデルを呼ぶ前まで */
export function prepareDurations(vb: VoicebankWeb, phrase: Phrase, spk: number): DurPrep {
  const ps = phoneSet(vb);
  const ph2 = splitPhraseSyllables(phrase, ps);
  if (!ph2.notes.length) return { phrase, ph2, feeds: null, where: [] };
  const { phs, durs, vals, where } = flatten(ph2, ps);
  const recs = extractPhraseRecords(phs, durs, vals, Math.trunc(spk), 0, ps).filter((r) => where.has(r.idx));
  if (!recs.length) return { phrase, ph2, feeds: null, where: [] };
  const inputs = (vb.cons_dur.inputs as string[]);
  return { phrase, ph2, feeds: recordsToFeeds(recs, inputs, ps), where: recs.map((r) => where.get(r.idx)!) };
}

// ── ② 長さの後処理（duration_postprocess.postprocess_durations） ──

interface Syl {
  onset: string[];
  nucleus: string[];
  coda: string[];
  pitch_midi: number;
  dur_sec: number;
  gap_after_sec: number;
  onset_nat: number[];
  coda_nat: number[];
  core_nat: number[];
  onset_dur: number[];
  coda_dur: number[];
  melisma: boolean; // PNote.melisma（音符の最初のまとまりだけ）
}

/** _floor_each: 各子音を自然長と下限の大きい方に（lead なら句の頭の下限） */
function floorEach(cons: string[], nat: number[], prof: Profile, lead = false): number[] {
  const out: number[] = [];
  const m = Math.min(cons.length, nat.length); // zip と同じく短い方まで
  for (let i = 0; i < m; i++) {
    const fl = lead ? prof.leadCMin(cons[i]) : prof.cMin(cons[i]);
    out.push(Math.max(nat[i], fl));
  }
  return out;
}

/** _scale_to_total: 比を保ったまま合計を total に */
function scaleToTotal(nat: number[], total: number): number[] {
  const s = pySum(nat);
  if (s <= 0.0) {
    const k = nat.length;
    return k ? new Array<number>(k).fill(total / k) : [];
  }
  return nat.map((x) => (x / s) * total);
}

/** _lead_pau_cons: 句の頭の onset と、その前の余白から (leading pau, onset の合計) */
function leadPauCons(consNat: number, space: number, pauTarget: number): [number, number] {
  const c = Math.max(0.0, consNat);
  const sp = Math.max(0.0, space);
  if (sp <= 0.0) return [0.0, 0.0];
  const cons = Math.min(c, sp);
  return [Math.min(pauTarget, sp - cons), cons];
}

/** _shrink_floored: 合計を cap にする。各要素は floor 以上を保つ（水位法） */
function shrinkFloored(nat: number[], floor: number[], cap: number): number[] {
  const n = nat.length;
  const out = nat.slice();
  const fixed = new Array<boolean>(n).fill(false);
  for (let it = 0; it < n + 1; it++) {
    const free: number[] = [];
    for (let i = 0; i < n; i++) if (!fixed[i]) free.push(i);
    if (!free.length) break;
    let fixedSum = 0;
    for (let i = 0; i < n; i++) if (fixed[i]) fixedSum += out[i];
    const budget = cap - fixedSum;
    let tot = 0;
    for (const i of free) tot += out[i];
    if (tot <= budget || tot <= 0) break;
    const s = budget / tot;
    let hit = false;
    for (const i of free) {
      const v = out[i] * s;
      if (v < floor[i]) {
        out[i] = floor[i]; fixed[i] = true; hit = true;
      } else {
        out[i] = v;
      }
    }
    if (!hit) break;
  }
  return out;
}

/** _shrink_to: 区間の中の子音（coda 全部 + onset の食い込み分）の合計を cap まで縮める（floor は割らない） */
function shrinkTo(codaNat: number[], onsetNextNat: number[], codaFloor: number[], onsetNextFloor: number[],
                  gap: number, overlap: number, cap: number): [number[], number[]] {
  if (overlap <= 0.0) return [shrinkFloored(codaNat, codaFloor, cap), onsetNextNat.slice()];
  const nat = codaNat.concat([overlap]);
  const flo = codaFloor.concat([Math.max(0.0, pySum(onsetNextFloor) - gap)]);
  const out = shrinkFloored(nat, flo, cap);
  const codaDur = out.slice(0, codaNat.length);
  const overlapFinal = out[out.length - 1];
  return [codaDur, scaleToTotal(onsetNextNat, gap + overlapFinal)];
}

/** _shrink: coda と（食い込みがあるときだけ）onset を係数 s で縮める */
function shrink(codaNat: number[], onsetNextNat: number[], gap: number, overlap: number,
                s: number): [number[], number[]] {
  const codaDur = codaNat.map((c) => c * s);
  if (overlap > 0) return [codaDur, scaleToTotal(onsetNextNat, gap + overlap * s)];
  return [codaDur, onsetNextNat.slice()];
}

/** _compress_region: 音符 j の区間を奪い合う coda_j と onset_(j+1) を確定の長さへ */
function compressRegion(dur: number, gap: number, codaNat: number[], onsetNextNat: number[], prof: Profile,
                        codaFloor: number[], onsetNextFloor: number[], codaWeight: number[],
                        onsetNextWeight: number[], nucleusWeight: number, vMin: number): [number[], number[]] {
  const oNext = pySum(onsetNextNat);
  const overlap = Math.max(0.0, oNext - gap);
  const codaTotal = pySum(codaNat);
  const demand = codaTotal + overlap;
  if (demand <= 0.0) return [codaNat.slice(), onsetNextNat.slice()];

  if (prof.enabled === false) {
    // floor 無効: 母音を食い尽くさない頭打ちだけ
    const room = Math.max(0.0, dur - NUC_EPS);
    if (demand <= room) return [codaNat.slice(), onsetNextNat.slice()];
    return shrink(codaNat, onsetNextNat, gap, overlap, room / demand);
  }

  const vm = vMin;
  const codaB = codaFloor.slice();
  const onsetB = onsetNextFloor.slice();
  const overlapB = Math.max(0.0, pySum(onsetB) - gap);
  const floorTotal = pySum(codaB) + overlapB;

  if (floorTotal + vm <= dur) {
    // (1) floor を保証できる: 子音は自然長、ただし母音に v_min を残す所まで
    const cap = dur - vm;
    if (demand <= cap) return [codaNat.slice(), onsetNextNat.slice()];
    return shrinkTo(codaNat, onsetNextNat, codaB, onsetB, gap, overlap, cap);
  }

  // (2) floor が入らない: 区間に居る音素（子音と母音）の重みの比で分ける
  const nC = codaNat.length;
  const nO = overlap > 0.0 ? onsetNextNat.length : 0;
  if (nC + nO <= 0) return [codaNat.slice(), onsetNextNat.slice()];
  const D = DEFAULT_PHONEME_WEIGHT;
  const pad = (w: number[], len: number) => w.concat(new Array<number>(len).fill(D)).slice(0, len);
  const wc = pad(codaWeight, nC);
  const wo = pad(onsetNextWeight, onsetNextNat.length);
  const wv = Math.max(0.05, nucleusWeight);
  const unit = Math.max(0.0, dur) / (pySum(wc) + pySum(wo.slice(0, nO)) + wv);
  const codaDur = wc.map((w) => unit * w);
  if (nO > 0) {
    // 次の onset は gap ぶんは只で使えるので、区間から取るのは重みぶんだけ
    const inside = unit * pySum(wo);
    return [codaDur, scaleToTotal(wo.map((w) => unit * w), gap + inside)];
  }
  return [codaDur, onsetNextNat.slice()];
}

/** _nucleus_weight: 区間の母音側の重み（核が複数音素なら平均） */
function nucleusWeight(prof: Profile, nucleus: string[]): number {
  const phs = nucleus.filter((v) => v !== PAU);
  if (!phs.length) return DEFAULT_PHONEME_WEIGHT;
  return pySum(phs.map((v) => prof.weight(v))) / phs.length;
}

/** _to_syls: 音符 → まとまり（自然長は最初のまとまりの onset・最後のまとまりの coda にだけ当てる） */
function toSyls(phrase: Phrase, ps: PhoneSet, onsetNat: number[][] | null, codaNat: number[][] | null,
                coreNat: number[][] | null): Syl[] {
  const syls: Syl[] = [];
  phrase.notes.forEach((note, k) => {
    const groups = syllabify(note.phonemes, ps);
    const m = groups.length;
    groups.forEach((g, gi) => {
      const [onset, nucleus, coda] = splitSyllable(g, ps);
      let on = onsetNat !== null && gi === 0 ? onsetNat[k] : null;
      let cd = codaNat !== null && gi === m - 1 ? codaNat[k] : null;
      if (on == null || on.length !== onset.length) on = new Array<number>(onset.length).fill(FALLBACK_CONS_DUR);
      if (cd == null || cd.length !== coda.length) cd = new Array<number>(coda.length).fill(FALLBACK_CONS_DUR);
      let co = coreNat !== null && m === 1 ? coreNat[k] : null;
      if (co == null || co.length !== nucleus.length) co = [];
      syls.push({
        onset, nucleus, coda, pitch_midi: note.pitch_midi, dur_sec: note.dur_sec / m,
        gap_after_sec: gi === m - 1 ? note.gap_after_sec : 0.0,
        onset_nat: on.slice(), coda_nat: cd.slice(), core_nat: co.slice(), onset_dur: [], coda_dur: [],
        melisma: !!note.melisma && gi === 0,
      });
    });
  });
  return syls;
}

/** _build_sequence: まとまりの列 → 平らな音素の列（dur_sec は float32） */
function buildSequence(syls: Syl[], leadingPau: number, trailingPau: number, ps: PhoneSet): PhonemeSequence {
  const phonemes: string[] = [];
  const durs: number[] = [];
  const midis: number[] = [];
  const vowels: number[] = [];
  const starts: number[] = [];
  const helds: number[] = [];
  const append = (ph: string, dur: number, midi: number, vow: boolean, start: boolean, held = false) => {
    phonemes.push(ph); durs.push(dur); midis.push(midi); vowels.push(vow ? 1 : 0); starts.push(start ? 1 : 0);
    helds.push(held ? 1 : 0);
  };
  append(PAU, leadingPau, 0, false, false);
  const n = syls.length;
  syls.forEach((s, j) => {
    s.onset.forEach((c, k) => append(c, s.onset_dur[k], s.pitch_midi, false, k === 0));
    const oNext = j + 1 < n ? pySum(syls[j + 1].onset_dur) : 0.0;
    const overlap = Math.max(0.0, oNext - s.gap_after_sec);
    const codaTotal = pySum(s.coda_dur);
    let effGap = s.gap_after_sec > 0 ? Math.max(0.0, s.gap_after_sec - oNext) : 0.0;
    let absorbed = 0.0;
    if (effGap > 0.0 && effGap < PAU_DROP_SEC) {
      absorbed = effGap;
      effGap = 0.0;
    }
    const nucTotal = Math.max(NUC_EPS, s.dur_sec - overlap - codaTotal) + absorbed;
    if (s.nucleus.length) {
      // 子音だけの音符は、自然長が分かっていればその比で分ける。分からなければ等分
      const coreOnly = !s.nucleus.some((v) => ps.isNucleus(v) || v === PAU);
      let w: number[] | null = null;
      if (coreOnly && s.core_nat.length && s.core_nat.length === s.nucleus.length) w = scaleToTotal(s.core_nat, nucTotal);
      const per = nucTotal / s.nucleus.length;
      let first = true;
      s.nucleus.forEach((v, iv) => {
        const dv = w !== null ? w[iv] : per;
        if (v === PAU) append(PAU, dv, 0, false, false);
        else {
          const isStart = first && !s.onset.length; // onset が無ければ最初の母音が音符の頭
          // 伸ばしの音符の最初の核が、すぐ前の音素と同じ母音なら「前の母音の続き」
          const held = isStart && s.melisma && ps.isNucleus(v) && phonemes.length > 0 && phonemes[phonemes.length - 1] === v;
          append(v, dv, s.pitch_midi, ps.isNucleus(v), isStart, held);
        }
        first = false;
      });
    } else {
      append(PAU, nucTotal, 0, false, !s.onset.length); // 母音の無い音符（退化した場合）は pau で埋める
    }
    s.coda.forEach((c, k) => append(c, s.coda_dur[k], s.pitch_midi, false, false));
    if (effGap > 0) append(PAU, effGap, 0, false, false);
  });
  append(PAU, trailingPau, 0, false, false);
  return {
    phonemes, dur_sec: Float32Array.from(durs), note_midi: Int32Array.from(midis),
    is_vowel: Uint8Array.from(vowels), note_event_starts: Uint8Array.from(starts), leading_pau_sec: leadingPau,
    held: Uint8Array.from(helds),
  };
}

/**
 * postprocess_durations: 句 → 音素の列と各音素の秒数（floor と圧縮を当てたもの）。
 * analyze と同じく rest_before = 1e9（句を独立に扱う）、style_id = 0。
 */
export function postprocessDurations(phrase: Phrase, prof: Profile, onsetNat: number[][] | null,
                                     codaNat: number[][] | null, coreNat: number[][] | null, ps: PhoneSet,
                                     leadingPauTarget = 0.25, restBefore = 1e9): PhonemeSequence {
  const syls = toSyls(phrase, ps, onsetNat, codaNat, coreNat);
  const n = syls.length;
  // 子音だけの音符の自然長にも下限を当てる（onset / coda と同じく floor → 尺合わせの順）
  for (const x of syls) {
    if (x.core_nat.length && !x.nucleus.some((v) => isCore(ps, v))) x.core_nat = floorEach(x.nucleus, x.core_nat, prof);
  }
  const codaFloored = syls.map((x) => floorEach(x.coda, x.coda_nat, prof));
  // 先頭のまとまりの onset だけは句の頭の下限（lead_c_min）
  const onsetFloored = syls.map((x, i) => floorEach(x.onset, x.onset_nat, prof, i === 0));
  const codaFloors = syls.map((x) => x.coda.map((c) => prof.cMin(c)));
  const onsetFloors = syls.map((x, i) => x.onset.map((c) => (i === 0 ? prof.leadCMin(c) : prof.cMin(c))));

  for (let j = 0; j < n; j++) {
    const nx = j + 1 < n;
    const [codaDur, onsetNextDur] = compressRegion(
      syls[j].dur_sec, syls[j].gap_after_sec, codaFloored[j], nx ? onsetFloored[j + 1] : [], prof,
      codaFloors[j], nx ? onsetFloors[j + 1] : [],
      syls[j].coda.map((c) => prof.weight(c)), (nx ? syls[j + 1].onset : []).map((c) => prof.weight(c)),
      nucleusWeight(prof, syls[j].nucleus), prof.vMin());
    syls[j].coda_dur = codaDur;
    if (nx) syls[j + 1].onset_dur = onsetNextDur;
  }

  // 先頭の onset と leading pau の交渉（onset_0 をここで確定）
  let leadingPau = phrase.leading_pau_sec;
  if (n > 0 && syls[0].onset.length) {
    const o0f = onsetFloored[0];
    const [lp, o0] = leadPauCons(pySum(o0f), restBefore, leadingPauTarget);
    leadingPau = lp;
    syls[0].onset_dur = scaleToTotal(o0f, o0);
  }
  return buildSequence(syls, leadingPau, phrase.trailing_pau_sec, ps);
}

// ── 句を置く位置・手で動かした分・フレームへの丸め ──

/** render.phrase_start_sec: 句の先頭の音符の核が楽譜の時刻に来る置き場所と、先頭の onset の長さ（負にもなる） */
export function phraseStartSec(phrase: Phrase, seq: PhonemeSequence, ps: PhoneSet): [number, number] {
  const nOn0 = phrase.notes.length ? splitSyllable(phrase.notes[0].phonemes, ps)[0].length : 0;
  const d = seq.dur_sec;
  // dur_sec は float32 なので、和も float32（numpy の sum）
  const onset0 = nOn0 ? npSum(d, Math.min(1, d.length), Math.min(1 + nOn0, d.length), true) : 0.0;
  const lead = d.length ? d[0] : 0.0;
  return [phrase.start_sec - lead - onset0, onset0];
}

/**
 * pipeline._apply_shifts: 手で動かした音素の頭を自然な長さに当てる（頭を s 秒ずらすと、前の音素が s 延び、自分が s 縮む）。
 * 前が pau の音素は、pau が pauMin 秒より短くならない所まで左へ動かせる。seq.dur_sec を float32 で書き換え、
 * 音素ごとに実際に当てたずれ（秒）を返す。
 */
export function applyShifts(seq: PhonemeSequence, noteIndex: number[],
                            shifts: Record<number, Record<number, number>>, pauMin: number): number[] {
  const d = Array.from(seq.dur_sec);
  const applied = new Array<number>(d.length).fill(0.0);
  const count = new Map<number, number>();
  noteIndex.forEach((ni, j) => {
    if (ni < 0) return;
    const idx = count.get(ni) ?? 0;
    count.set(ni, idx + 1);
    const per = shifts[ni];
    let s = per ? per[idx] : undefined;
    // Python の `if not s`: 無い・0 なら動かさない
    if (s === undefined || s === null || s === 0 || j === 0) return;
    const lo = seq.phonemes[j - 1] === PAU ? Math.min(-(d[j - 1] - pauMin), 0.0) : -(d[j - 1] - MIN_PH_SEC);
    s = Math.min(Math.max(s, lo), d[j] - MIN_PH_SEC); // np.clip と同じ（lo > hi なら hi）
    d[j - 1] += s;
    d[j] -= s;
    applied[j] = s;
  });
  seq.dur_sec = Float32Array.from(d);
  return applied;
}

/** duration_postprocess.to_frames: 秒 → 音素ごとのフレーム数（境目ごとに丸める）と総フレーム数 */
export function toFrames(durSec: Float32Array, frameRate: number): [Int32Array, number] {
  const n = durSec.length;
  const bnd = new Array<number>(n + 1);
  bnd[0] = 0;
  let acc = 0; // np.cumsum は float32 のまま足していく
  for (let i = 0; i < n; i++) {
    acc = Math.fround(acc + durSec[i]);
    bnd[i + 1] = roundHalfEven(acc * frameRate);
  }
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.max(bnd[i + 1] - bnd[i], 1);
  return [out, bnd[n]];
}

/** render._align_durs: 音素ごとのフレーム数の合計を T に合わせる（最後の音素で増減。各要素は 1 以上） */
export function alignDurs(phDurs: Int32Array, T: number): Int32Array {
  const out = Int32Array.from(phDurs);
  const diff = T - out.reduce((a, b) => a + b, 0);
  if (diff !== 0 && out.length > 0) out[out.length - 1] = Math.max(1, out[out.length - 1] + diff);
  return out;
}

// ── ① の後半〜②: 自然長を当て、後処理から音素の番号の対応・手で動かした分・フレームまで ──

export interface DurResult {
  ph2: Phrase;
  /** postprocess_durations に渡した自然長（ph2 の音符ごと） */
  naturals: { onset: number[][]; coda: number[][]; core: number[][] };
  /** 手で動かした後の音素の列 */
  seq: PhonemeSequence;
  /** 手で動かす前の長さ（float32） */
  seq0DurSec: Float32Array;
  /** 句の頭（先頭の pau の頭）を置く絶対時刻（手で動かす前の長さで決める） */
  startSec: number;
  /** 先頭の onset の長さ [s] */
  onset0Sec: number;
  /** 音素ごとの、元の句の中の音符の番号（pau は -1） */
  noteIndex: number[];
  /** 音素ごとに実際に当てた頭のずれ [s] */
  applied: number[];
  /** 音素ごとのフレーム数（to_frames の出力。F0 の長さに合わせる前） */
  phDurs: Int32Array;
  totalFrames: number;
}

/**
 * PrettyPitchLeapSinger._cons_naturals の残り（モデルの出力を自然長として当てる）と analyze の ② まで。
 * pred: cons_dur.onnx の dur_sec（prep.feeds が null なら null でよい）。
 * shifts: {句の中の音符の番号: {音符の中の音素の番号: 秒}}（手で動かした音素の頭）。
 */
export function finishDurations(vb: VoicebankWeb, prep: DurPrep, pred: Float32Array | null, spk: number,
                                shifts: Record<number, Record<number, number>>): DurResult {
  const ps = phoneSet(vb);
  const { phrase, ph2 } = prep;
  const onset: number[][] = [];
  const coda: number[][] = [];
  const core: number[][] = [];
  for (const note of ph2.notes) {
    const [on, nuc, cd] = splitSyllable(note.phonemes, ps);
    const coreOnly = nuc.length > 0 && !nuc.some((v) => isCore(ps, v));
    onset.push(new Array<number>(on.length).fill(FALLBACK_CONS_DUR));
    coda.push(new Array<number>(cd.length).fill(FALLBACK_CONS_DUR));
    core.push(coreOnly ? new Array<number>(nuc.length).fill(FALLBACK_CONS_DUR) : []);
  }
  if (pred && prep.feeds) {
    const slots = { onset, coda, core };
    const m = Math.min(prep.where.length, pred.length); // zip と同じく短い方まで
    for (let r = 0; r < m; r++) {
      const [j, slot, k] = prep.where[r];
      slots[slot][j][k] = pred[r];
    }
  }
  const prof = new Profile(vb, spk);
  const seq = postprocessDurations(ph2, prof, onset, coda, core, ps, 0.25, 1e9);
  const seq0DurSec = Float32Array.from(seq.dur_sec);
  // 句を置く位置は、手で動かす前の長さで決める
  const [startSec, onset0Sec] = phraseStartSec(ph2, seq, ps);
  // 音素 → 元の句の中の音符の番号。音節の頭を数え、核が複数ある音符（分けた後の音符）は元の音符へ戻す
  let syl2note: number[] = [];
  phrase.notes.forEach((n, j) => {
    const m = syllabify(n.phonemes, ps).length;
    for (let q = 0; q < m; q++) syl2note.push(j);
  });
  if (syl2note.length !== ph2.notes.length) syl2note = ph2.notes.map((_, i) => i);
  const noteIndex: number[] = [];
  let k = -1;
  seq.phonemes.forEach((ph, i) => {
    if (seq.note_event_starts[i]) k++;
    noteIndex.push(ph === PAU ? -1 : syl2note[Math.min(Math.max(k, 0), syl2note.length - 1)]);
  });
  if (vb.acoustic_merge_melisma && shifts) {
    // 音響モデルで前の母音とまとめる音素（held）は、画面でも前の音素につなげて見せ、頭を動かせないので、
    // 残っていたずれは当てない（engine/pipeline.py の analyze と同じ）
    const cnt = new Map<number, number>();
    const drop = new Set<string>();
    noteIndex.forEach((ni, i) => {
      if (ni < 0) return;
      cnt.set(ni, (cnt.get(ni) ?? 0) + 1);
      if (seq.held[i]) drop.add(`${ni}:${cnt.get(ni)! - 1}`);
    });
    if (drop.size) {
      const kept: Record<number, Record<number, number>> = {};
      for (const [ni, d] of Object.entries(shifts)) {
        kept[Number(ni)] = Object.fromEntries(Object.entries(d).filter(([k]) => !drop.has(`${ni}:${k}`)));
      }
      shifts = kept;
    }
  }
  const pauMin = PAU_MIN_FRAMES / vb.frame_rate + PAU_MIN_EPS;
  const applied = applyShifts(seq, noteIndex, shifts ?? {}, pauMin);
  const [phDurs, totalFrames] = toFrames(seq.dur_sec, vb.frame_rate);
  return { ph2, naturals: { onset, coda, core }, seq, seq0DurSec, startSec, onset0Sec, noteIndex, applied,
           phDurs, totalFrames };
}
