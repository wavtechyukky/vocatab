// 再生（モックの仮の音）: 各トラックのピッチの線をなぞるのこぎり波を、音量の曲線で鳴らす。
// エンジンをつないだら、ここは句ごとに作った本物の音を並べて鳴らす形に替える。
import { audioContext, readyAudioContext } from "./audio";
import { audioBuffer } from "./audiofiles";
import { buildFxChain, type FxChain } from "./fx";
import { derive } from "./derive";
import { t as tx } from "./i18n";
import { FPS, isAudioTrack, secPerTick, type Track, trackRegion } from "./model";
import { evalCurve, paramDef } from "./params";
import type { Store } from "./store";
import type { View } from "./view";
import { KEY_W } from "./view";

/** AudioParam に入れる値: 有限でなければ代わりの値（NaN を入れると例外になり、再生の準備が止まる） */
const finite = (v: number, fallback: number): number => (Number.isFinite(v) ? v : fallback);

export class Transport {
  playing = false;
  follow = true;
  private starting = false; // AudioContext を起こしている間（二重に再生しないように）
  private nodes: AudioScheduledSourceNode[] = [];
  private startTick = 0; // 再生を始めた位置（停止で戻る）
  private t0 = 0; // 再生を始めた AudioContext の時刻
  private ctx: AudioContext | null = null; // 再生に使っている AudioContext（途中で作り直されても時刻はこれで測る）
  private raf = 0;

  /** トラックの合成済みの波形（曲の頭から、44.1 kHz）。無ければ仮の音で鳴らす。 */
  engineWav: ((track: Track) => Float32Array | null) | null = null;
  /** 再生の準備で起きた失敗を知らせる（画面の下の欄に出す） */
  onError: (message: string) => void = (m) => console.error(m);
  /** トラックの id → 再生中の音量とパンのノード（つまみや M・S を動かすとその場で変わる） */
  private mixNodes = new Map<number, { gain: GainNode; pan: StereoPannerNode; fx: FxChain | null }>();

  constructor(private store: Store, private view: View, private viewWidth: () => number,
              private onState: () => void) {
    store.on((kind) => {
      if (kind === "mix" || kind === "project") this.updateMix();
    });
  }

  /** トラックの出口（エフェクト → 音量 → パン → スピーカー）を作る。伴奏はエフェクトを通さない（ステレオのまま）。 */
  private trackOut(a: AudioContext, track: Track): AudioNode {
    const fx = isAudioTrack(track) ? null : buildFxChain(a);
    const gain = a.createGain();
    const pan = a.createStereoPanner();
    fx?.output.connect(gain);
    gain.connect(pan).connect(a.destination);
    this.mixNodes.set(track.id, { gain, pan, fx });
    this.updateMix();
    return fx ? fx.input : gain;
  }

  /** トラックの音量・パン・消音・ソロを、鳴っている音に反映する。 */
  updateMix(): void {
    const tracks = this.store.project.tracks;
    const anySolo = tracks.some((t) => t.solo);
    for (const t of tracks) {
      const m = this.mixNodes.get(t.id);
      if (!m) continue;
      const audible = !t.mute && (!anySolo || t.solo);
      const db = finite(t.gain ?? 0, 0);
      m.gain.gain.value = audible && db > -24 ? 10 ** (db / 20) : 0; // つまみの左端（−24 dB）は無音
      m.pan.pan.value = Math.max(-1, Math.min(1, finite(t.pan ?? 0, 0)));
      try {
        m.fx?.update(t.fx);
      } catch (e) {
        this.onError(tx("「{name}」のエフェクトを掛けられません: {e}", { name: t.name, e: e instanceof Error ? e.message : String(e) }));
      }
    }
  }

  toggle(): void {
    if (this.playing) this.pause();
    else void this.play();
  }

  async play(): Promise<void> {
    if (this.playing || this.starting) return;
    this.starting = true;
    let a: AudioContext;
    try {
      a = await readyAudioContext();
    } finally {
      this.starting = false;
    }
    if (this.playing) return;
    this.ctx = a;
    const s = this.store;
    const spt = secPerTick(s.project.bpm);
    this.startTick = s.playTick;
    const startSec = s.playTick * spt;
    const when = a.currentTime + 0.05;
    // 消音・ソロで聞こえないトラックも鳴らしておき、音量 0 にする（再生中に M・S を切り替えられるように）。
    // 1 つのトラックの準備が失敗しても、ほかのトラックは鳴らす（失敗は下の欄に出す）
    for (const track of s.project.tracks) {
      try {
        this.startTrack(a, track, when, startSec, spt);
      } catch (e) {
        this.onError(tx("「{name}」を鳴らせません: {e}", { name: track.name, e: e instanceof Error ? e.message : String(e) }));
      }
    }
    this.playing = true;
    this.t0 = when;
    this.tick(startSec);
    this.onState();
  }

  /** 1 つのトラックの音を、AudioContext の時刻 when（曲の startSec 秒の所）から鳴らす。 */
  private startTrack(a: AudioContext, track: Track, when: number, startSec: number, spt: number): void {
    const s = this.store;
    if (isAudioTrack(track)) {
      // 伴奏: 範囲の頭から、音声の offsetSec の所を鳴らす
      const buf = track.audio && audioBuffer(track.audio.id);
      if (!buf || !track.audio) return;
      const r = trackRegion(track, s.project.beatsPerBar);
      const r0 = r.start * spt;
      const r1 = r.end * spt;
      if (startSec >= r1) return;
      const node = a.createBufferSource();
      node.buffer = buf;
      node.connect(this.trackOut(a, track));
      const lead = Math.max(0, r0 - startSec); // 再生の頭から伴奏が始まるまで
      const into = finite(track.audio.offsetSec, 0) + Math.max(0, startSec - r0); // 音声の中のどこから
      node.start(when + lead, into, Math.max(0, r1 - Math.max(r0, startSec)));
      this.nodes.push(node);
      return;
    }
    if (!track.notes.length) return;
    const wav = this.engineWav?.(track) ?? null;
    if (wav) {
      // エンジンの音: 句を並べた波形をそのまま鳴らす
      const off = Math.floor(startSec * 44100);
      if (off >= wav.length) return;
      const buf = a.createBuffer(1, wav.length - off, 44100);
      buf.copyToChannel(wav.subarray(off), 0);
      const src = a.createBufferSource();
      src.buffer = buf;
      src.connect(this.trackOut(a, track));
      src.start(when);
      this.nodes.push(src);
      return;
    }
    const der = derive(s.project, track);
    const i0 = Math.max(0, Math.floor(startSec * FPS));
    const n = der.pitch.length - i0;
    if (n < 2) return;
    const freq = new Float32Array(n);
    const gain = new Float32Array(n);
    let last = 220;
    const ld = paramDef("loudness").def;
    for (let k = 0; k < n; k++) {
      const p = der.pitch[i0 + k];
      const voiced = Number.isFinite(p);
      if (voiced) last = 440 * 2 ** ((p - 69) / 12);
      freq[k] = last;
      const db = track.base.loudness + evalCurve(track.params.loudness, (i0 + k) / FPS / spt, ld);
      gain[k] = voiced ? finite(0.09 * 10 ** (db / 20), 0) : 0;
    }
    const dur = n / FPS;
    const osc = a.createOscillator();
    osc.type = "sawtooth";
    osc.frequency.setValueCurveAtTime(freq, when, dur);
    const lp = a.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 1800;
    lp.Q.value = 0.7;
    const g = a.createGain();
    g.gain.setValueAtTime(0, when);
    g.gain.setValueCurveAtTime(gain, when, dur);
    osc.connect(lp).connect(g).connect(this.trackOut(a, track));
    osc.start(when);
    osc.stop(when + dur);
    this.nodes.push(osc);
  }

  private tick(startSec: number): void {
    const a = this.ctx ?? audioContext();
    const s = this.store;
    const spt = secPerTick(s.project.bpm);
    const sec = startSec + Math.max(0, a.currentTime - this.t0);
    s.playTick = sec / spt;
    s.changed("playhead");
    if (this.follow) {
      // 再生位置が右の端に近づいたら、画面を送る
      const x = this.view.tickToX(s.playTick);
      if (x > this.viewWidth() - 40 || x < KEY_W) {
        this.view.scrollTick = Math.max(0, s.playTick - 40 / this.view.pxPerTick);
        this.view.emit();
      }
    }
    let end = 0;
    for (const t of s.project.tracks) {
      if (isAudioTrack(t)) end = Math.max(end, trackRegion(t, s.project.beatsPerBar).end);
      for (const n of t.notes) end = Math.max(end, n.onset + n.dur);
    }
    if (s.playTick > end + 480) {
      this.stop();
      return;
    }
    this.raf = requestAnimationFrame(() => this.tick(startSec));
  }

  private halt(): void {
    cancelAnimationFrame(this.raf);
    for (const n of this.nodes) {
      try {
        n.stop();
      } catch {
        // すでに止まっている
      }
    }
    this.nodes = [];
    for (const m of this.mixNodes.values()) m.pan.disconnect();
    this.mixNodes.clear();
    this.playing = false;
  }

  /**
   * 再生中に合成し直した音を、今の再生位置から鳴らし直す。再生は始めたときの波形を写して鳴らすので、
   * これを呼ばないと、再生中の変更は次に再生するまで聞こえない。停止で戻る位置（再生を始めた位置）はそのまま。
   */
  async refresh(): Promise<void> {
    if (!this.playing || this.starting) return;
    const keepStart = this.startTick;
    this.halt();
    await this.play();
    this.startTick = keepStart;
  }

  /** 一時停止: 再生位置はその場に残す。 */
  pause(): void {
    this.halt();
    this.onState();
  }

  /** 停止: 再生を始めた位置に戻る（止まっているときは頭に戻る）。 */
  stop(): void {
    const wasPlaying = this.playing;
    this.halt();
    this.store.playTick = wasPlaying ? this.startTick : 0;
    this.store.changed("playhead");
    this.onState();
  }
}
