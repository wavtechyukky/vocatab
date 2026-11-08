// 自動のレンダリング: 編集のたびに（少し待ってから）フレーズごとに解析し、最後の F0 が変わったフレーズだけを合成し直す。
// ブレス（音符の前に入れた息継ぎ）はエンジンから波形をもらい、画面側で「音量」の曲線を掛けてフレーズの波形に混ぜる。
// エンジンは engine-web/client.ts の窓口（既定はブラウザの中。Python のサーバーが配ったページや ?engine=server では手元の Python）。
import { type BreathClipInfo, type BreathSlot, breathPart } from "./breath";
import type { BreathContext } from "./derive";
import { composeF0, type EngineInfo, type EPhrase, hashF32, phraseParams, singerDisplay } from "./engine";
import { type EngineClient, makeEngine } from "./engine-web/client";
import { wavToMel } from "./engine-web/melspec";
import type { RestoreEntry } from "./engine-web/protocol";
import { t as tx } from "./i18n";
import { activeNotes, FPS, type Track } from "./model";
import type { Store } from "./store";

const N_MELS = 128;
const MEL_FLOOR = -11.5;

export type MelKind = "in" | "out" | "plain";

/** 曲の時間軸に並べる mel の 1 片（フレーズか、ブレス）。startFrame は絶対のフレーム、mel は bin ごと [128 × frames] */
export interface MelPiece {
  startFrame: number;
  frames: number;
  mel: Float32Array;
}

/**
 * mel の片を曲の時間軸に並べる（bin ごと [128 × N]）。重なる所（句の端の pau・ブレス）は、音は波形で足しているので
 * mel もパワーの和にする: ln-mel は振幅の対数なので 0.5·ln(e^{2a} + e^{2b})（床の所はそのまま置く）
 */
export function assembleMel(pieces: MelPiece[], endFrame: number): Float32Array | null {
  if (!pieces.length) return null;
  let n = endFrame;
  for (const p of pieces) n = Math.max(n, p.startFrame + p.frames);
  n += 172;
  const out = new Float32Array(N_MELS * n).fill(MEL_FLOOR);
  for (const p of pieces) {
    for (let b = 0; b < N_MELS; b++) {
      for (let i = 0; i < p.frames; i++) {
        const g = p.startFrame + i;
        if (g < 0 || g >= n) continue;
        const k = b * n + g;
        const v = p.mel[b * p.frames + i];
        const cur = out[k];
        if (cur <= MEL_FLOOR) out[k] = Math.max(cur, v);
        else if (v > MEL_FLOOR) out[k] = 0.5 * Math.log(Math.exp(2 * cur) + Math.exp(2 * v));
      }
    }
  }
  return out;
}

/** 句の mel を片に（無い句は飛ばす） */
export function phraseMelPieces(phrases: EPhrase[], kind: MelKind): MelPiece[] {
  const pick = (p: EPhrase) => (kind === "in" ? p.melIn : kind === "plain" ? p.melPlain : p.melOut);
  return phrases.flatMap((p) => (pick(p) ? [{ startFrame: p.startFrame, frames: p.nFrames, mel: pick(p)! }] : []));
}

/** 波形の部品（フレーズとブレス。start は絶対のサンプル位置）を曲の時間軸に並べる（44.1 kHz）。部品が無ければ null。 */
export function assembleParts(parts: Array<{ start: number; wav: Float32Array }>): Float32Array | null {
  if (!parts.length) return null;
  let n = 0;
  for (const p of parts) n = Math.max(n, p.start + p.wav.length);
  const out = new Float32Array(Math.max(1, n));
  for (const p of parts) {
    const w = p.wav;
    for (let i = 0; i < w.length; i++) {
      const g = p.start + i;
      if (g >= 0 && g < n) out[g] += w[i];
    }
  }
  return out;
}

/** 有限でないサンプル（NaN・無限大）を 0 にして、その数を返す。1 つでも残すと、再生のときトラックのフィルタ
 *  （低域カット・イコライザ）の中身が NaN になり、そこから先がずっと無音になる（合成し直すまで直らない） */
function scrubWav(w: Float32Array): number {
  let n = 0;
  for (let i = 0; i < w.length; i++) {
    if (!Number.isFinite(w[i])) {
      w[i] = 0;
      n++;
    }
  }
  return n;
}

export class Renderer {
  info: EngineInfo | null = null;
  /** トラックの id → 句の列（最新の解析） */
  tracks = new Map<number, EPhrase[]>();
  rev = 0; // 句の中身が変わるたびに増える（表示の取り置きの鍵）
  error = "";
  /** mel の ② の行がオンか（オンのときだけ、ボイスパラメータを掛けない音の mel も頼む） */
  wantPlain: () => boolean = () => false;
  /** mel の ③ の行がオンか（オンのときだけ、出た音の mel を頼む。1 句で 35 ms ほど） */
  wantOut: () => boolean = () => false;
  /** トラックのブレスの置き場所（derive の結果。main.ts が入れる。無ければブレスは混ぜない） */
  slotsFor: (track: Track) => BreathSlot[] = () => [];
  /** 編集の後の合成が全部済んで、合成し直した句があったとき（再生中なら鳴らし直すため。main.ts が入れる） */
  onSynthesized: () => void = () => {};
  readonly engine: EngineClient = makeEngine();
  private busy = false;
  private timer = 0;
  private breathIndex = new Map<string, BreathClipInfo[]>(); // ボイスバンク → ブレスの束の一覧（届いたもの）
  private breathIndexPending = new Set<string>();
  /** "ボイスバンク|ボコーダ|番号" → ブレスの波形（取りに行っている間は null） */
  private breathWavs = new Map<string, { wav: Float32Array } | null>();

  constructor(private store: Store, private onUpdate: () => void) {
    this.engine.onProgress = () => this.onUpdate(); // モデルの読み込みの進み具合
    this.engine.onState = () => this.onUpdate(); // 待機に入った・起きた
    // 待機から起きたエンジンへ、画面が持っている解析の結果を戻す（中身の変わっていない句を解析し直さないため）
    this.engine.collectRestore = () => {
      if (!this.info) return [];
      const out: RestoreEntry[] = [];
      for (const track of this.store.project.tracks) {
        const vb = this.voicebankOf(track).key;
        for (const ph of this.phrases(track)) {
          out.push({ key: ph.key, vb, singer: this.singerId(track), ids: ph.noteIds, phonemes: ph.phonemes,
                     start_sec: ph.startSec, plain: ph.plain, vib: ph.vib, uv: ph.uv, base: ph.base, seed: ph.seed });
        }
      }
      return out;
    };
  }

  async init(): Promise<void> {
    this.info = await this.engine.info();
    this.onUpdate();
    if (this.info) {
      this.schedule(0);
      for (const tr of this.store.project.tracks) this.breathContext(tr); // ブレスの束を先に取りに行く（小さい）
    }
  }

  get connected(): boolean {
    return this.info !== null && this.info.voicebanks.length > 0;
  }

  /** 選んでいるトラックのボイスバンク（状態の表示に使う） */
  get voicebank() {
    return this.voicebankOf(this.store.track);
  }

  /** トラックが歌うボイスバンク（key が一覧に無ければ先頭） */
  voicebankOf(track: Track) {
    const vbs = this.info!.voicebanks;
    return vbs.find((v) => v.key === track.voicebank) ?? vbs[0];
  }

  singerId(track: Track): number {
    return this.voicebankOf(track).singers.find((s) => s.name === track.voice)?.id ?? 0;
  }

  /** トラックの歌い手の表示名（言語に合わせたフルネーム。つながる前は内部の名前） */
  singerLabel(track: Track): string {
    const s = this.connected ? this.voicebankOf(track).singers.find((x) => x.name === track.voice) : undefined;
    return singerDisplay(s, track.voice);
  }

  /** 選んでいるボコーダの鍵（NSF-HiFiGAN が無ければ NHVSing を使う）。 */
  vocoderKey(): string {
    const vocs = this.info?.vocoders ?? [];
    const want = this.store.vocoder === "nsf" ? "nsf" : "nhvsing";
    return (vocs.find((v) => v.kind.includes(want)) ?? vocs[0])?.key ?? "";
  }

  hasVocoder(kind: "nhv" | "nsf"): boolean {
    return (this.info?.vocoders ?? []).some((v) => v.kind.includes(kind === "nsf" ? "nsf" : "nhvsing"));
  }

  phrases(track: Track): EPhrase[] {
    return this.tracks.get(track.id) ?? [];
  }

  /** まだ合成が済んでいない句の数（書き出す前の確認に使う）。 */
  pending(): number {
    if (this.busy) return Math.max(1, this.countPending());
    return this.countPending();
  }

  private countPending(): number {
    let n = 0;
    for (const t of this.store.project.tracks) {
      for (const ph of this.phrases(t)) if (ph.status !== "済み") n++;
      for (const s of this.slotsFor(t)) if (s.index !== null && !this.breathWav(t, s.index)) n++; // 無ければ取りに行く
    }
    return n;
  }

  // ── ブレス ──

  /** トラックのブレスの文脈（derive に渡す）。束をまだ読んでいなければ取りに行き、届いたら知らせる。無ければ undefined */
  breathContext(track: Track): BreathContext | undefined {
    if (!this.connected) return undefined;
    const vb = this.voicebankOf(track).key;
    const clips = this.breathIndex.get(vb);
    if (!clips) {
      if (!this.breathIndexPending.has(vb)) {
        this.breathIndexPending.add(vb);
        this.engine.breaths(vb).then((r) => {
          this.breathIndex.set(vb, r.clips);
          this.rev++;
          this.onUpdate();
        }).catch((e) => {
          console.warn(e);
          this.breathIndex.set(vb, []); // 読めなければブレス無しとして進む
        }).finally(() => this.breathIndexPending.delete(vb));
      }
      return undefined;
    }
    return clips.length ? { clips, singer: this.singerId(track) } : undefined;
  }

  /** ブレスの束の様子（表示用）: 取りに行っている途中 / 無い / 有る */
  breathState(track: Track): "loading" | "none" | "ready" {
    if (!this.connected) return "none";
    const clips = this.breathIndex.get(this.voicebankOf(track).key);
    if (!clips) return "loading";
    return clips.some((c) => c.singer === this.singerId(track)) ? "ready" : "none";
  }

  /** ブレスの波形（ボイスバンク・ボコーダ・番号ごとに取り置く）。無ければエンジンに頼んで null を返し、届いたら知らせる */
  breathWav(track: Track, index: number): { wav: Float32Array } | null {
    const vb = this.voicebankOf(track).key;
    const voc = this.vocoderKey();
    const key = `${vb}|${voc}|${index}`;
    const hit = this.breathWavs.get(key);
    if (hit !== undefined) return hit;
    this.breathWavs.set(key, null);
    this.engine.breath({ voicebank: vb, index, vocoder: voc }).then((r) => {
      scrubWav(r.wav);
      this.breathWavs.set(key, { wav: r.wav });
      this.rev++;
      this.onUpdate();
    }).catch((e) => {
      this.breathWavs.delete(key);
      this.error = tx("ブレスの合成に失敗: {e}", { e: e instanceof Error ? e.message : String(e) });
      this.onUpdate();
    });
    return null;
  }

  /** トラックの波形の部品（合成済みのフレーズと、届いているブレス）。start は絶対のサンプル位置 */
  trackParts(track: Track, slots: BreathSlot[] = this.slotsFor(track)): Array<{ start: number; wav: Float32Array }> {
    const parts = this.phrases(track).flatMap((p) => (p.wav ? [{ start: p.startFrame * 256, wav: p.wav }] : []));
    for (const s of slots) {
      if (s.index === null) continue;
      const b = this.breathWav(track, s.index);
      if (b) parts.push(breathPart(s, b.wav, track, this.store.project));
    }
    return parts;
  }

  /** トラックのエフェクト前の波形（再生・書き出し用）。合成済みのものが無ければ null */
  trackWav(track: Track): Float32Array | null {
    return assembleParts(this.trackParts(track));
  }

  /**
   * トラックの mel（画面の mel のパネル用）。ブレスはフレーズのボコーダを通らない（別に鳴らして波形で足す）ので、
   * ①（ボコーダの入力）と ②（ボコーダの出力）には出さず、③（声の調整後 = 実際に鳴る音）にだけ、
   * 音量の曲線を掛けた後のブレスの音から計算した mel を重ねる
   */
  trackMel(track: Track, kind: MelKind): Float32Array | null {
    const phrases = this.phrases(track);
    const pieces = phraseMelPieces(phrases, kind);
    if (!pieces.length) return null; // 句の mel が 1 つも無ければ、ブレスだけでは出さない
    let end = 0;
    for (const p of phrases) end = Math.max(end, p.startFrame + p.nFrames);
    if (kind === "out") {
      for (const s of this.slotsFor(track)) {
        if (s.index === null) continue;
        const b = this.breathWav(track, s.index);
        if (!b) continue;
        const part = breathPart(s, b.wav, track, this.store.project);
        const { mel, frames } = wavToMel(part.wav);
        pieces.push({ startFrame: Math.round((s.start + s.cut) * FPS), frames, mel });
      }
    }
    return assembleMel(pieces, end);
  }

  schedule(delay = 200): void {
    if (!this.connected) return;
    clearTimeout(this.timer);
    this.timer = window.setTimeout(() => void this.run(), delay);
  }

  private async run(): Promise<void> {
    if (this.busy) {
      this.schedule(100); // 走っている途中の編集は、終わってからもう一度
      return;
    }
    this.busy = true;
    const ver = this.store.version;
    const p = this.store.project;
    let made = 0; // この回に合成し直した句の数
    try {
      // 1. 解析（速い。中身が同じ句はエンジン側で取っておいたものを返す）
      for (const track of p.tracks) {
        // トラックの範囲で切った音符だけを歌わせる（端をまたぐ音符は端で終わる）
        const notes = activeNotes(track, p.beatsPerBar);
        if (!notes.length) {
          this.tracks.set(track.id, []);
          this.rev++;
          continue;
        }
        let j;
        try {
          j = await this.engine.analyze({
            voicebank: this.voicebankOf(track).key, singer: this.singerId(track), bpm: p.bpm, seed: 0,
            notes: notes.map((n) => ({ id: n.id, onset: n.onset, dur: n.dur, pitch: n.pitch, lyric: n.lyric,
                                       retake_f0: n.attrs.retakeF0 ?? 0, ph_shift: n.phShift ?? {} })),
            lyric_style: p.lyricStyle ?? "sequential", // 英語の + の読み方（ほかの言語では使わない）
          });
        } catch (e) {
          this.error = tx("解析に失敗: {e}", { e: e instanceof Error ? e.message : String(e) });
          this.onUpdate();
          continue;
        }
        this.error = "";
        // 中身の同じ句（鍵が同じ）は合成の結果（波形と 3 種の mel）を引き継ぐ。② を落とすと、② の行がオンのとき
        // 編集のたびに全部の句を合成し直してしまう
        const old = new Map(this.phrases(track).map((q) => [q.key, q]));
        const list: EPhrase[] = j.phrases.map((q) => {
          const prev = old.get(q.key);
          return {
            key: q.key, noteIds: q.note_ids, startFrame: q.start_frame, startSec: q.start_sec, nFrames: q.n_frames,
            phonemes: q.phonemes, plain: q.plain, vib: q.vib, uv: q.uv, base: q.base, seed: q.seed,
            status: prev?.status ?? "解析済み", f0Hash: prev?.f0Hash, wav: prev?.wav, melIn: prev?.melIn,
            melOut: prev?.melOut, melPlain: prev?.melPlain,
          };
        });
        this.tracks.set(track.id, list);
        this.rev++;
        this.onUpdate();
      }
      // 2. 合成（最後の F0・歌い手・ボコーダが変わった句だけ）
      const voc = this.vocoderKey();
      for (const track of p.tracks) {
        for (const ph of this.phrases(track)) {
          if (this.store.version !== ver) return; // 編集が入った: 最初からやり直す（finally で予約）
          const singer = this.singerId(track);
          const { log2 } = composeF0(ph, track, p);
          const params = phraseParams(ph, track, p);
          const ps = Object.keys(params).sort().map((k) => hashF32(params[k], k)).join(",");
          const plain = this.wantPlain();
          const out = this.wantOut();
          // AI リテイク（声）: この句の音符のうち取り直したものの種
          const byId = new Map(track.notes.map((n) => [n.id, n]));
          const retakeMel: Record<string, number> = {};
          // 動かした子音の凹み: この句の音符のうち、凹みを動かしたもの
          const phDip: Record<string, Record<string, { db?: number; w?: number }>> = {};
          for (const id of ph.noteIds) {
            const s = byId.get(id)?.attrs.retakeMel;
            if (s) retakeMel[String(id)] = s;
            const d = byId.get(id)?.phDip;
            if (d && Object.keys(d).length) phDip[String(id)] = d;
          }
          const h = hashF32(log2, `${voc}|${singer}|${ps}|${JSON.stringify(retakeMel)}|${JSON.stringify(phDip)}`);
          // ②・③ の行をオンにしたばかりで mel が無い句は、同じ音でも頼み直す（エンジンの取り置きに当たるので速い）
          if (ph.f0Hash === h && ph.wav && (!plain || ph.melPlain) && (!out || ph.melOut)) continue;
          ph.status = "合成中";
          this.onUpdate();
          let j;
          try {
            // 句の鍵にボイスバンクが入っているので、ボイスバンクを替えた句は解析から作り直しになる
            j = await this.engine.synthesize({ voicebank: this.voicebankOf(track).key, singer, key: ph.key, f0_log2: log2, vocoder: voc, params,
                                               mel_out: out, mel_plain: plain, retake_mel: retakeMel, ph_dip: phDip });
          } catch (e) {
            ph.status = "失敗";
            this.error = tx("合成に失敗: {e}", { e: e instanceof Error ? e.message : String(e) });
            this.onUpdate();
            continue;
          }
          ph.wav = j.wav;
          const bad = scrubWav(ph.wav);
          if (bad) {
            const sec = (ph.startFrame / (44100 / 256)).toFixed(1);
            console.warn(`句（${sec} 秒〜、鍵 ${ph.key}）の波形に有限でない値が ${bad} 個`);
            this.error = tx("{sec} 秒からのフレーズに不正な値が {n} 個ありました（無音に置き換えて再生します）", { sec, n: bad });
          }
          ph.melIn = j.mel_in;
          ph.melOut = j.mel_out;
          ph.melPlain = j.mel_plain;
          ph.f0Hash = h;
          ph.status = "済み";
          made++;
          this.rev++;
          this.onUpdate();
        }
      }
    } catch (e) {
      this.error = tx("エンジンとの通信に失敗: {e}", { e: String(e) });
      this.onUpdate();
    } finally {
      this.busy = false;
      if (this.store.version !== ver) this.schedule(50);
      else {
        this.onUpdate(); // 全部済んだことを知らせる（書き出しのボタンなど）
        if (made) this.onSynthesized();
      }
    }
  }
}
