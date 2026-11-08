// ブラウザ版のエンジンの本体。engine/server.py・engine/pipeline.py と同じ順で、
// 楽譜 → 音素の長さ → F0（analyze）、最後の F0 → mel → 波形（synthesize）を行う。モデルは onnxruntime-web で動かす。
// ふだんは Web Worker（worker.ts）の中で動き、照らし合わせの試験（ui/test/e2e）ではページの中で直接動かす。
import { alignDurs, finishDurations, prepareDurations } from "./durations";
import { f0Feeds, finishF0, frameNoteIndex, prepareF0 } from "./f0host";
import { fetchCached } from "./fetchcache";
import { wavToMel } from "./melspec";
import type { AnalyzeReq, AnalyzeResp, BreathIndexResp, BreathReq, BreathResp, RestoreEntry, SynthReq, SynthResp } from "./protocol";
import { applyRetakes, normalNoise } from "./rng";
import { phrasesFromNotes } from "./score";
import { acousticFeeds, finishSynth, fromFloat16Bits, melPostprocess, nhvFeeds, nsfFeeds, phoneSetOf, toFloat16Bits,
         type VocOut } from "./synthhost";
import { f32 } from "./types";
import type { BreathBank, Feeds, Phrase, VoicebankWeb, VocoderWeb, WebIndex } from "./types";

const MEL_FLOOR = Math.fround(Math.log(1e-5)); // mel の床（pau・詰め物）
const N_MELS = 128;
// ブレス（engine/pipeline.py の BREATH_F0・BREATH_PAD・BREATH_RAMP_SEC と同じ）
const BREATH_F0 = 220.0; // 全フレーム無声なので調波は出ない。0 を避けるためのダミー
const BREATH_PAD = 4; // 前後に足す床のフレーム（ボコーダのフィルタの立ち上がりを吸収し、後で切る）
const BREATH_RAMP_SEC = 0.05; // 両端の Tukey 窓の長さ（短い素材では長さの 1/4）

/** Tukey 窓の値（0〜1）: 頭と尻尾を ramp 秒ずつ余弦で立ち上げ・下げる。sec は素材の中の時刻、len は長さ（秒） */
export function tukey(sec: number, len: number, ramp = BREATH_RAMP_SEC): number {
  const r = Math.min(ramp, len / 4);
  if (r <= 0) return 1;
  const d = Math.min(sec, len - sec);
  return d >= r ? 1 : d <= 0 ? 0 : 0.5 * (1 - Math.cos((Math.PI * d) / r));
}

// ── 取り置き ──

/** 個数で区切る取り置き（engine/pipeline.py の _LRU と同じ）。解析の結果用（1 つが小さい） */
class LRU<V> {
  private d = new Map<string, V>();
  constructor(private size: number) {}
  has(k: string): boolean {
    return this.d.has(k);
  }
  get(k: string): V | undefined {
    const v = this.d.get(k);
    if (v !== undefined) {
      this.d.delete(k);
      this.d.set(k, v);
    }
    return v;
  }
  put(k: string, v: V): void {
    this.d.delete(k);
    this.d.set(k, v);
    while (this.d.size > this.size) this.d.delete(this.d.keys().next().value as string);
  }
}

/**
 * 合計のバイト数で区切る取り置き。音響・ボコーダの結果は句の長さで 1 つの大きさが何倍も違うので、個数ではなくバイト数で区切る。
 * さらに tag（解析の鍵 = 句）ごとに新しい方から perTag 個だけ残す: AI リテイクや F0 の編集のたびに同じ句の変種が増えるが、
 * 戻るとしても一つ前までなので、それより古い種のものは手放す（リテイクを連打しても取り置きが膨らまない）
 */
export class ByteLRU<V> {
  private d = new Map<string, { v: V; bytes: number; tag: string }>();
  private total = 0;
  constructor(private cap: number, private perTag = Infinity) {}

  /** 今の合計のバイト数 */
  get bytes(): number {
    return this.total;
  }

  get size(): number {
    return this.d.size;
  }

  get(k: string): V | undefined {
    const e = this.d.get(k);
    if (e === undefined) return undefined;
    this.d.delete(k);
    this.d.set(k, e);
    return e.v;
  }

  put(k: string, v: V, bytes: number, tag: string): void {
    this.drop(k);
    this.d.set(k, { v, bytes, tag });
    this.total += bytes;
    // 同じ句の古い変種を手放す（新しい方から perTag 個を残す）
    let n = 0;
    for (const kk of [...this.d.keys()].reverse()) {
      if (this.d.get(kk)!.tag !== tag) continue;
      if (++n > this.perTag) this.drop(kk);
    }
    // 合計が上限を超えたら古い順に手放す（入れたばかりのものは上限より大きくても残す）
    for (const kk of this.d.keys()) {
      if (this.total <= this.cap || kk === k) break;
      this.drop(kk);
    }
  }

  private drop(k: string): void {
    const e = this.d.get(k);
    if (e === undefined) return;
    this.d.delete(k);
    this.total -= e.bytes;
  }
}

/** 値の並びの指紋（FNV-1a を 2 本。取り置きの鍵） */
function digest(...parts: unknown[]): string {
  let h1 = 0x811c9dc5, h2 = 0x01000193 ^ 0x5bd1e995;
  const feed = (x: number) => {
    h1 = Math.imul(h1 ^ x, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ x, 0x5bd1e995) >>> 0;
  };
  for (const p of parts) {
    if (ArrayBuffer.isView(p)) {
      const v = p as ArrayBufferView;
      const u = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
      for (let i = 0; i < u.length; i++) feed(u[i]);
    } else {
      const s = JSON.stringify(p, (_, x) => (ArrayBuffer.isView(x) ? Array.from(x as unknown as ArrayLike<number>) : x));
      for (let i = 0; i < s.length; i++) feed(s.charCodeAt(i));
    }
    feed(0x7c);
  }
  return h1.toString(16).padStart(8, "0") + h2.toString(16).padStart(8, "0");
}

/** Python の round（ちょうど半分は偶数へ） */
export const roundHalfEven = (x: number): number => {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
};

interface Analysis {
  phonemes: string[];
  phDurs: Int32Array;
  startSec: number;
  plainLog2: Float32Array;
  vibLog2: Float32Array;
  uv: Float32Array;
  baseLog2: Float32Array;
  noteIndex: number[];
  seed: number;
  shifts: number[];
}

/** ⑤⑥ の結果（取り置き） */
interface AcousticHit {
  melIn: Float32Array;
  uvHard: Float32Array;
  cons: Uint8Array;
  f0: Float32Array;
}

/**
 * 照らし合わせの試験用: 乱数の代わりに使う雑音（Python 版と同じ雑音を入れて、同じ結果になるかを見る）。
 * 返した配列はそのまま使う（AI リテイクの差し替えもしない。Python の雑音には差し替え済みのものを渡す）
 */
export interface NoiseHook {
  f0?(T: number): Float32Array | null;
  exc?(T: number): Float32Array | null;
  voc?(T: number): Float32Array | null;
}

/** 取り置きの上限（init の前に変えられる。既定は Python 版の取り置きと同じくらいの量） */
export interface CacheLimits {
  /** ⑤⑥ の結果（mel ①など）。7.8 秒の句で 1 つ約 0.7 MB */
  acCacheBytes?: number;
  /** ⑦ の結果（調波・雑音・float16 の包絡）。7.8 秒の句で 1 つ約 5.5 MB */
  vocCacheBytes?: number;
  /** 同じ句の変種（リテイクの種・F0）を新しい方からいくつ残すか */
  variantsPerPhrase?: number;
}

/**
 * モデルごとの計算の場所。GPU（WebGPU）が使えるときだけ、精度が CPU と変わらず速くなるものを GPU で動かす
 * （research/experiment/web_ort_bench の計測。7.8 秒の句で、F0 は 364 → 60 ms、NSF-HiFiGAN は 8.4 → 1.2 秒、
 * 出力の差は CPU のときと同じ程度）。音響モデルは GPU だと出力のずれが CPU の約 9 倍（mel の平均 0.01）になり、
 * NHVSing は GPU のほうが遅いので、どちらも CPU（WASM）で動かす
 */
const GPU_MODELS = new Set(["f0", "nsf_hifigan"]);

/** ボコーダの取り置き 1 つのバイト数 */
const vocBytes = (vo: VocOut): number =>
  (vo.harmonic?.byteLength ?? 0) + (vo.noise_out?.byteLength ?? 0) + (vo.env_h?.byteLength ?? 0) +
  (vo.env_n?.byteLength ?? 0) + (vo.waveform?.byteLength ?? 0);

export class Engine {
  private base = "";
  private gpu = false; // WebGPU が使えるか（init で調べる）
  private creating: Promise<unknown> = Promise.resolve(); // セッション作りを 1 つずつにする鎖
  private index!: WebIndex;
  private vbs = new Map<string, VoicebankWeb>();
  private sessions = new Map<string, Promise<any>>();
  private analyses = new LRU<{ vb: string; an: Analysis; ids: number[] }>(256);
  private acCache: ByteLRU<AcousticHit>;
  private vocCache: ByteLRU<VocOut>;
  private breathBanks = new Map<string, Promise<{ idx: BreathBank; data: Uint16Array } | null>>(); // ボイスバンク → ブレスの束
  private loading = new Map<string, { loaded: number; total: number }>();
  noise: NoiseHook = {};

  /** ort: onnxruntime-web。onProgress: モデルの読み込みの進み具合（全部の合計）。limits: 取り置きの上限 */
  constructor(private ort: any, private onProgress: (loaded: number, total: number) => void = () => undefined,
              limits: CacheLimits = {}) {
    const per = limits.variantsPerPhrase ?? 3; // 元の取り置きと、聴き比べ用のリテイク 2 つ
    this.acCache = new ByteLRU(limits.acCacheBytes ?? 32 * 1024 * 1024, per);
    this.vocCache = new ByteLRU(limits.vocCacheBytes ?? 128 * 1024 * 1024, per);
  }

  /** base: リポジトリの頭の URL（web_index.json・voicebanks・vocoders がある所）。useGpu は頼まれたときだけ（既定は CPU） */
  async init(base: string, preload = true, useGpu = false): Promise<WebIndex> {
    this.base = base;
    try {
      const g = (globalThis.navigator as any)?.gpu;
      this.gpu = useGpu && !!g && !!(await g.requestAdapter()) && !!this.ort.env?.webgpu;
    } catch {
      this.gpu = false;
    }
    // 設定の JSON は毎回サーバーに新しいか確かめる（cache: "no-cache"）。ブラウザが古い web.json を使い回すと、
    // 足したばかりの項目（ブレスの束など）が見えない。中身が同じなら 304 で済む
    this.index = await (await fetch(this.url("web_index.json"), { cache: "no-cache" })).json();
    for (const v of this.index.voicebanks) this.vbs.set(v.key, await (await fetch(this.url(v.web), { cache: "no-cache" })).json());
    if (preload) {
      // 解析に使う小さいモデル（子音の長さ・F0）を先に、合成の大きいモデルを後ろで読み始める（ボコーダは既定のものだけ）
      for (const v of this.index.voicebanks) {
        void this.session(v.key, "cons_dur").catch(() => undefined);
        void this.session(v.key, "f0").catch(() => undefined);
      }
      for (const v of this.index.voicebanks) {
        void this.session(v.key, "acoustic").catch(() => undefined);
        const voc = this.index.vocoders.find((x) => x.key === v.default_vocoder);
        if (voc) void this.vocoderSession(voc).catch(() => undefined);
      }
    }
    return this.index;
  }

  private url(rel: string): string {
    return new URL(rel, this.base).href;
  }

  private progress(key: string, loaded: number, total: number): void {
    this.loading.set(key, { loaded, total });
    let l = 0, t = 0;
    for (const v of this.loading.values()) {
      l += v.loaded;
      t += v.total;
    }
    this.onProgress(l, t);
  }

  /** 計算の場所の名前（試験・表示用）: "webgpu" か "wasm" */
  device(kind: string): string {
    return this.gpu && GPU_MODELS.has(kind) ? "webgpu" : "wasm";
  }

  /** 取り置きの様子（確認用）: 合計のバイト数と個数 */
  cacheStats(): { acBytes: number; acCount: number; vocBytes: number; vocCount: number } {
    return { acBytes: this.acCache.bytes, acCount: this.acCache.size, vocBytes: this.vocCache.bytes, vocCount: this.vocCache.size };
  }

  private load(key: string, rel: string, bytes: number, kind: string): Promise<any> {
    let s = this.sessions.get(key);
    if (!s) {
      const p = (async () => {
        const data = await fetchCached(this.url(rel), bytes, (l, t) => this.progress(key, l, t));
        // セッション作りは 1 つずつ（同時に作ると onnxruntime-web の初期化が 2 回走り、GPU の準備が失敗する）
        const create = async (): Promise<any> => {
          const opts = { graphOptimizationLevel: "all", enableMemPattern: false };
          if (this.device(kind) === "webgpu") {
            try {
              return await this.ort.InferenceSession.create(data, { ...opts, executionProviders: ["webgpu"] });
            } catch (e) {
              console.warn(`GPU で準備できなかったので CPU で動かす（${kind}）`, e);
            }
          }
          return this.ort.InferenceSession.create(data, { ...opts, executionProviders: ["wasm"] });
        };
        const c = this.creating.then(create, create);
        this.creating = c.catch(() => undefined);
        return c;
      })();
      // 失敗したら、次に頼まれたときに読み直す（手放した後に同じ鍵で作り直したものは消さない）
      p.catch(() => {
        if (this.sessions.get(key) === p) this.sessions.delete(key);
      });
      this.sessions.set(key, p);
      s = p;
    }
    return s;
  }

  private session(vbKey: string, model: "cons_dur" | "f0" | "acoustic"): Promise<any> {
    const vi = this.index.voicebanks.find((v) => v.key === vbKey)!;
    return this.load(`${vbKey}/${model}`, this.vbs.get(vbKey)!.models[model], vi.bytes[model], model);
  }

  private vocoderSession(v: VocoderWeb): Promise<any> {
    return this.load(`voc/${v.key}`, v.onnx, v.bytes, v.kind);
  }

  /**
   * 使うボコーダ以外のセッションを手放す（NHVSing と NSF-HiFiGAN を両方持つと WASM のメモリが 60 MB ほど余計に要る。
   * 読み直しは Cache Storage からなので速い）。画面は合成を 1 句ずつ順に頼むので、手放す相手で計算の途中ということは無い
   */
  private releaseVocodersExcept(key: string): void {
    for (const [k, p] of [...this.sessions]) {
      if (!k.startsWith("voc/") || k === `voc/${key}`) continue;
      this.sessions.delete(k);
      this.loading.delete(k);
      p.then((s: any) => s.release()).catch(() => undefined);
    }
  }

  private async run(sess: any, feeds: Feeds, names?: string[]): Promise<Record<string, Float32Array>> {
    const inputs: Record<string, any> = {};
    for (const [k, f] of Object.entries(feeds)) inputs[k] = new this.ort.Tensor(f.type, f.data, f.dims);
    const out = names ? await sess.run(inputs, names) : await sess.run(inputs);
    const r: Record<string, Float32Array> = {};
    for (const [k, t] of Object.entries(out)) r[k] = (t as any).data as Float32Array;
    return r;
  }

  // ── 解析（engine/server.py の analyze と PrettyPitchLeapSinger.analyze） ──
  async analyze(r: AnalyzeReq): Promise<AnalyzeResp> {
    const vb = this.vbs.get(r.voicebank);
    if (!vb) throw new Error(`ボイスバンクがありません: ${r.voicebank}`);
    const fr = vb.frame_rate;
    const phrases = phrasesFromNotes(vb, r.notes, r.bpm);
    const retake = new Map(r.notes.map((n) => [n.id, n.retake_f0 ?? 0]));
    const shift = new Map(r.notes.map((n) => [n.id, Object.fromEntries(Object.entries(n.ph_shift ?? {})
      .filter(([, v]) => v).map(([k, v]) => [Number(k), Number(v)]))]));
    const out: AnalyzeResp["phrases"] = [];
    for (const { phrase, ids } of phrases) {
      // 句の中の音符の番号ごとの取り直しの種・手で動かした音素の頭（engine/server.py と同じ）
      const f0Rt: Record<number, number> = {};
      const sh: Record<number, Record<number, number>> = {};
      ids.forEach((nid, i) => {
        if (retake.get(nid)) f0Rt[i] = retake.get(nid)!;
        const s = shift.get(nid);
        if (s && Object.keys(s).length) sh[i] = s;
      });
      const key = digest(r.voicebank, r.singer, r.seed, phrase, f0Rt, sh);
      let hit = this.analyses.get(key);
      if (!hit) {
        hit = { vb: r.voicebank, an: await this.analyzePhrase(vb, phrase, r.singer, r.seed, f0Rt, sh), ids };
        this.analyses.put(key, hit);
      }
      const an = hit.an;
      const ps = phoneSetOf(vb);
      out.push({
        key, note_ids: ids, start_frame: roundHalfEven(an.startSec * fr), start_sec: an.startSec,
        n_frames: an.phDurs.reduce((a, b) => a + b, 0),
        phonemes: an.phonemes.map((p, j) => ({ ph: p, frames: an.phDurs[j], shift: an.shifts[j] ?? 0,
          note: an.noteIndex[j] >= 0 && ids.length ? ids[Math.min(an.noteIndex[j], ids.length - 1)] : null,
          nucleus: ps.isNucleus(p) })), // 画面の色分け（母音など）
        plain: an.plainLog2, vib: an.vibLog2, uv: an.uv, base: an.baseLog2, seed: an.seed,
      });
    }
    return { frame_rate: fr, phrases: out };
  }

  private async analyzePhrase(vb: VoicebankWeb, phrase: Phrase, spk: number, seed: number,
                              f0Rt: Record<number, number>, sh: Record<number, Record<number, number>>): Promise<Analysis> {
    // ① 子音の自然長 → ② 長さの後処理（と手で動かした頭）
    const prep = prepareDurations(vb, phrase, spk);
    const pred = prep.feeds ? (await this.run(await this.session(vb.key, "cons_dur"), prep.feeds, ["dur_sec"])).dur_sec : null;
    const dur = finishDurations(vb, prep, pred, spk, sh);
    // ③ F0（拡散の雑音は句の種から。取り直した音符の所だけ別の種の雑音） → ④ ピッチ補正
    const fp = prepareF0(vb, dur.seq, spk);
    let noise = this.noise.f0?.(fp.T) ?? null;
    if (!noise) {
      noise = normalNoise(Number(vb.f0.n_noise) * 2 * fp.T, seed);
      if (Object.keys(f0Rt).length && dur.noteIndex.length === fp.phDurs.length) {
        applyRetakes(noise, frameNoteIndex(dur.noteIndex, fp.phDurs, fp.T), f0Rt, 3);
      }
    }
    const o = await this.run(await this.session(vb.key, "f0"), f0Feeds(vb, fp, spk, noise), ["plain", "vibrato", "vuv"]);
    const f = finishF0(vb, fp, { plain: o.plain, vibrato: o.vibrato, vuv: o.vuv });
    return { phonemes: dur.seq.phonemes, phDurs: alignDurs(dur.phDurs, f.T), startSec: dur.startSec,
             plainLog2: f.plainLog2, vibLog2: f.vibLog2, uv: f.uv, baseLog2: f.baseLog2,
             noteIndex: dur.noteIndex, seed, shifts: dur.applied };
  }

  /**
   * 待機から起きたとき（client.ts の wake）: 画面が持っている解析の結果を取り置きへ戻す。
   * analyze が返した形から Analysis を組み立て直す（音素のフレーム数 → phDurs、音符の id → 句の中の番号、shift → shifts）ので、
   * 中身の変わっていない句は解析し直さずに済み、合成の鍵（種・音素・長さ）も前と同じになる。戻した数を返す
   */
  restore(entries: RestoreEntry[]): number {
    let n = 0;
    for (const e of entries) {
      if (!this.vbs.has(e.vb) || this.analyses.has(e.key)) continue;
      const an: Analysis = {
        phonemes: e.phonemes.map((p) => p.ph),
        phDurs: Int32Array.from(e.phonemes, (p) => p.frames),
        startSec: e.start_sec,
        plainLog2: e.plain, vibLog2: e.vib, uv: e.uv, baseLog2: e.base,
        noteIndex: e.phonemes.map((p) => (p.note === null ? -1 : e.ids.indexOf(p.note))),
        seed: e.seed,
        shifts: e.phonemes.map((p) => p.shift ?? 0),
      };
      this.analyses.put(e.key, { vb: e.vb, an, ids: e.ids.slice() });
      n++;
    }
    return n;
  }

  // ── 合成（engine/server.py の synthesize と PrettyPitchLeapSinger.synthesize） ──
  async synthesize(r: SynthReq): Promise<SynthResp> {
    const hit = this.analyses.get(r.key);
    if (!hit) throw new Error("解析の結果がありません（解析からやり直してください）");
    const vb = this.vbs.get(hit.vb)!;
    const voc = this.index.vocoders.find((v) => v.key === r.vocoder);
    if (!voc) throw new Error(`ボコーダがありません: ${r.vocoder}`);
    this.releaseVocodersExcept(voc.key); // 使うボコーダだけを持つ
    const an = hit.an;
    const T = an.phDurs.reduce((a, b) => a + b, 0);
    // engine/server.py: f0_hz = 2 ** f0_log2（float64 で計算して float32 に）
    const f0Hz = new Float32Array(r.f0_log2.length);
    for (let i = 0; i < f0Hz.length; i++) f0Hz[i] = Math.pow(2, r.f0_log2[i]);
    // 声のリテイク: 音符の id → 句の中の番号
    const pos = new Map(hit.ids.map((nid, i) => [String(nid), i]));
    const rt: Record<number, number> = {};
    for (const [k, s] of Object.entries(r.retake_mel ?? {})) if (pos.has(k) && Number(s)) rt[pos.get(k)!] = Number(s);

    // ⑤ 音響 → ⑥ mel の後処理（F0・歌い手・取り直しが同じなら使い回す）
    const acKey = digest(an.seed, r.singer, an.phonemes, an.phDurs, f0Hz.subarray(0, T), rt);
    let ac = this.acCache.get(acKey);
    if (!ac) {
      let exc = this.noise.exc?.(T) ?? null;
      if (!exc) {
        exc = normalNoise(T * 256, an.seed, 1);
        if (Object.keys(rt).length) {
          const fn = frameNoteIndex(an.noteIndex, an.phDurs, T);
          const fs = new Int32Array(T * 256);
          for (let i = 0; i < fs.length; i++) fs[i] = fn[(i / 256) | 0];
          applyRetakes(exc, fs, rt, 1);
        }
      }
      const a = acousticFeeds(vb, { phonemes: an.phonemes, phDurs: an.phDurs, uv: an.uv, noteIndex: an.noteIndex }, f0Hz, r.singer, exc);
      const acSess = await this.session(vb.key, "acoustic");
      // 子音の凹みの入力（x0_loggain）を持つモデル: ブラウザ版のエンジンはまだ凹みを作らないので、0（凹みなし）を渡す
      if ((acSess.inputNames as string[]).includes("x0_loggain")) a.feeds.x0_loggain = f32(new Float32Array(T), [1, T]);
      const mel = (await this.run(acSess, a.feeds, ["mel"])).mel;
      ac = { melIn: melPostprocess(mel, T, a.uvHard, vb.frame_rate), uvHard: a.uvHard, cons: a.cons, f0: a.f0 };
      this.acCache.put(acKey, ac, ac.melIn.byteLength + ac.uvHard.byteLength + ac.cons.byteLength + ac.f0.byteLength, r.key);
    }
    // ⑦ ボコーダ（さらにボコーダが同じなら使い回す）。ボイスパラメータはこの後で掛けるので、動かしても混ぜ直すだけ
    const vocKey = digest(acKey, voc.key);
    let vo = this.vocCache.get(vocKey);
    if (!vo) {
      const sess = await this.vocoderSession(voc);
      if (voc.kind === "nhvsing") {
        const o = await this.run(sess, nhvFeeds(ac.melIn, T, ac.f0, ac.uvHard, this.noise.voc?.(T) ?? normalNoise(T * 256, an.seed, 2)));
        // 取り置きは Python（_vocode）と同じ形: 波形は持たず（調波 + 雑音から作り直す）、包絡は float16（ビット）で持つ。
        // 7.8 秒の句で 9.7 MB → 5.5 MB
        vo = { harmonic: o.harmonic, noise_out: o.noise_out,
               env_h: o.env_h ? toFloat16Bits(o.env_h) : undefined, env_n: o.env_n ? toFloat16Bits(o.env_n) : undefined };
      } else {
        vo = { waveform: (await this.run(sess, nsfFeeds(ac.melIn, T, ac.f0), ["waveform"])).waveform };
      }
      this.vocCache.put(vocKey, vo, vocBytes(vo), r.key);
    }
    const fin = finishSynth(voc, T, vo, r.params, ac.uvHard, ac.cons, r.mel_plain);
    // 出た音の mel（engine/server.py の mel_of: 足りなければ ln(1e-5) で埋めて T に揃える）
    const melOf = (w: Float32Array): Float32Array => {
      const { mel, frames } = wavToMel(w);
      const m = new Float32Array(N_MELS * T).fill(MEL_FLOOR);
      for (let b = 0; b < N_MELS; b++) for (let i = 0; i < Math.min(T, frames); i++) m[b * T + i] = mel[b * frames + i];
      return m;
    };
    return { n_frames: T, wav: fin.wav, mel_in: ac.melIn.slice(0, N_MELS * T),
             mel_out: r.mel_out ? melOf(fin.wav) : undefined,
             mel_plain: r.mel_plain && fin.wavPlain ? melOf(fin.wavPlain) : undefined };
  }

  // ── ブレス（engine/pipeline.py の load_breaths・breath_index・breath） ──

  /** ボイスバンクのブレスの束（breaths.json と float16 の bin）。無ければ null。初めて要るときに読む */
  private breathBank(vbKey: string): Promise<{ idx: BreathBank; data: Uint16Array } | null> {
    let p = this.breathBanks.get(vbKey);
    if (!p) {
      p = (async () => {
        const vb = this.vbs.get(vbKey);
        if (!vb?.breaths) return null;
        const r = await fetch(this.url(vb.breaths), { cache: "no-cache" });
        if (!r.ok) throw new Error(`ブレスの一覧を読めません: ${vb.breaths}（${r.status}）`);
        const idx = (await r.json()) as BreathBank;
        const binUrl = new URL(idx.bin, this.url(vb.breaths)).href;
        const bytes = await fetchCached(binUrl, idx.bin_bytes, () => undefined); // 小さいので進み具合は出さない
        return { idx, data: new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1) };
      })();
      p.catch(() => this.breathBanks.delete(vbKey)); // 失敗したら次に頼まれたとき読み直す
      this.breathBanks.set(vbKey, p);
    }
    return p;
  }

  async breaths(vbKey: string): Promise<BreathIndexResp> {
    if (!this.vbs.has(vbKey)) throw new Error(`ボイスバンクがありません: ${vbKey}`);
    const bank = await this.breathBank(vbKey);
    return { clips: (bank?.idx.clips ?? []).map((c) => ({ id: c.id, singer: c.singer, frames: c.frames, sec: c.sec })) };
  }

  /**
   * ブレス 1 つ → 波形 [256·frames]。ブレスはフレーズのボコーダには入れず（声の調整は音量だけ）、別に鳴らして画面が波形で足す。
   * mel に床の詰め物を付け、f0 一定・uv = 1（全フレーム無声）でボコーダを通し、声と同じ後段
   * （ボイスパラメータは既定値 → 調波 + 雑音とリミッタだけ）を経て詰め物を切り、両端に Tukey 窓を掛ける。
   * engine/pipeline.py の breath と同じ手順。mel の後処理（音量合わせ）は通さない: 全フレーム無声だと静かなブレスを持ち上げてしまうため
   */
  async breath(r: BreathReq): Promise<BreathResp> {
    const bank = await this.breathBank(r.voicebank);
    const c = bank?.idx.clips[r.index];
    if (!bank || !c) throw new Error(`ブレスがありません: ${r.voicebank} ${r.index}`);
    const voc = this.index.vocoders.find((v) => v.key === r.vocoder);
    if (!voc) throw new Error(`ボコーダがありません: ${r.vocoder}`);
    const F = c.frames;
    const T = F + 2 * BREATH_PAD;
    const raw = fromFloat16Bits(bank.data.subarray(c.offset, c.offset + N_MELS * F));
    const off = c.gain_db ? Math.fround(Math.log(10 ** (c.gain_db / 20))) : 0;
    const mel = new Float32Array(N_MELS * T).fill(MEL_FLOOR);
    for (let b = 0; b < N_MELS; b++) {
      for (let t = 0; t < F; t++) mel[b * T + BREATH_PAD + t] = off ? Math.fround(raw[b * F + t] + off) : raw[b * F + t];
    }
    const f0 = new Float32Array(T).fill(BREATH_F0);
    const uvHard = new Float32Array(T);
    const sess = await this.vocoderSession(voc);
    let vo: VocOut;
    if (voc.kind === "nhvsing") {
      const o = await this.run(sess, nhvFeeds(mel, T, f0, uvHard, this.noise.voc?.(T) ?? normalNoise(T * 256, 7000 + r.index, 2)));
      vo = { harmonic: o.harmonic, noise_out: o.noise_out, env_h: o.env_h, env_n: o.env_n };
    } else {
      vo = { waveform: (await this.run(sess, nsfFeeds(mel, T, f0), ["waveform"])).waveform };
    }
    const fin = finishSynth(voc, T, vo, {}, uvHard, new Uint8Array(T), false);
    const wav = fin.wav.slice(BREATH_PAD * 256, (BREATH_PAD + F) * 256);
    // 両端を Tukey 窓でなだらかに
    const len = F * 256 / 44100;
    for (let i = 0; i < wav.length; i++) wav[i] = Math.fround(wav[i] * tukey((i + 0.5) / 44100, len));
    return { frames: F, wav };
  }
}
