// 下のパネル: 最下段のタブで開け閉めし、開いたものはピアノロールの下に積み重なる。
// 「音素のタイミング」は音素の境界を動かすほか、ブレス（息継ぎ）を入れる所でもある: 入れられる休符には「＋ 息」の印が出て、
// クリックで入る（パターンは音符ごとに決まる）。入れてある所はクリックで別のパターンに、右クリックで消す。
// 子音の凹み（ボイスバンクが凹みを掛けるとき）: 子音の箱に、上の線（凹みなし）から垂れ下がる谷を描き、谷の底の頂点を
// 動かす。上下 = 深さ、左右 = 広さ（谷はいつも子音の頭から始まり、頂点は谷の真ん中）。ダブルクリックで既定に戻す。
// 空いている所を横にドラッグすると、その範囲の頂点を選ぶ（⌘ / Ctrl を押しながらだと足す。⌘ / Ctrl クリックで 1 つずつ足す・外す）。
// 選んだ頂点のどれかを上下に動かすと、選んだもの全部の深さが同じだけ変わる。範囲の中か選んだ頂点をダブルクリックで、まとめて既定に戻す。
import { addOrCycleBreath, type BreathSlot } from "./breath";
import { cssVar, HiDpiCanvas, requestDraw } from "./canvas";
import { type BreathContext, type Derived, DIP_DB_MIN, type PhoneSeg } from "./derive";
import { drawTimeGrid } from "./grid";
import { t } from "./i18n";
import { secPerTick } from "./model";
import type { Store } from "./store";
import { KEY_W, type View, wheelScrollX } from "./view";

// 子音の凹みの描き方。縦軸は箱の上の線（0 dB）から下へ DIP_DB_MIN（derive.ts）まで（変換表を入れるまでは名目の深さ）。
// 名目で −5 dB より深くすると、出力の mel はほぼ無音になる（2026-11-08 に聴いて確かめた）ので、そこを動かせる限界にする
const DIP_W_MIN = 0.1; // 広さの下限（子音の長さに対する割合）
const DIP_HANDLE_R = 2.5; // 頂点の半径（px）。曲線を隠さないよう小さく（つかめる距離は DIP_HIT_R）
const DIP_HIT_R = 8; // 頂点をつかめる距離（px）
const DIP_FINE = 0.25; // Shift を押しながら動かすときの倍率

const dipKey = (s: { noteId: number; idx: number }): string => `${s.noteId}:${s.idx}`;

/** 凹みの曲線（ν 字の谷）: 谷の中の位置 u（0〜1）での dB。底（u = 0.5）が db */
function dipCurveDb(db: number, u: number): number {
  const amp = Math.pow(10, db / 20);
  const g = 1 - (1 - amp) * (1 - Math.cos(2 * Math.PI * u)) / 2;
  return 20 * Math.log10(Math.max(g, 1e-4));
}

export class PhonemePanel {
  readonly hc: HiDpiCanvas;
  private drag: { noteId: number; idx: number; baseStart: number; before: string } | null = null;
  private dipDrag: { noteId: number; idx: number; len: number; x: number; y: number; db0: number; w0: number;
                     before: string } | null = null;
  private dipHover: PhoneSeg | null = null; // 値を出す凹み（頂点の上にマウスがあるか、動かしている途中）
  private sel = new Set<string>(); // 選んだ凹みの頂点（dipKey）
  private selRange: { a: number; b: number } | null = null; // 範囲で選んだときの時刻（秒）。中のクリックでは選択を外さない
  // 空いている所を押した: 横に動かせば範囲選択、動かさずに離せばクリック（ブレスの所ならブレス、範囲の外なら選択を外す）
  private band: { x0: number; x1: number; add: boolean; breath: BreathSlot | null; moved: boolean } | null = null;
  private groupDrag: { items: Array<{ noteId: number; idx: number; db0: number }>; y: number; before: string } | null = null;

  /** breathCtx: ブレスの束と歌い手（エンジンがつながっていて束があるときだけ。無ければブレスの所は出ない） */
  constructor(private store: Store, private view: View, private derived: () => Derived, canvas: HTMLCanvasElement,
              private snapTicks: () => number, private hint: (s: string) => void,
              private breathCtx: () => BreathContext | undefined = () => undefined) {
    this.hc = new HiDpiCanvas(canvas, () => this.draw());
    canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", () => this.onUp());
    canvas.addEventListener("pointerleave", () => {
      if (this.dipHover && !this.dipDrag) {
        this.dipHover = null;
        this.redraw();
      }
    });
    canvas.addEventListener("dblclick", (e) => this.onDbl(e));
    canvas.addEventListener("contextmenu", (e) => this.onContext(e));
    canvas.addEventListener("wheel", (e) => wheelScrollX(this.view, e), { passive: false });
  }

  redraw(): void {
    requestDraw(() => this.draw());
  }

  private secToX(sec: number): number {
    return this.view.tickToX(sec / secPerTick(this.store.project.bpm));
  }

  /** x の近くの、動かせる境界（音素の頭）。 */
  private boundaryAt(x: number): PhoneSeg | null {
    let best: PhoneSeg | null = null;
    let bd = 6;
    for (const s of this.derived().phones) {
      if (!s.movable) continue;
      const d = Math.abs(this.secToX(s.start) - x);
      if (d < bd) {
        bd = d;
        best = s;
      }
    }
    return best;
  }

  /** x の下のブレスの区画（入れてある所と、入れられる所）。 */
  private breathAt(x: number): BreathSlot | null {
    for (const b of this.derived().breaths) {
      if (x >= this.secToX(b.start + b.cut) && x <= this.secToX(b.end)) return b;
    }
    return null;
  }

  /** 凹みを描く帯の上（0 dB）と下（DIP_DB_MIN）の y。音素の名前の下から箱の下まで */
  private dipBand(): { y0: number; y1: number } {
    const h = this.hc.h;
    return { y0: 8 + 21, y1: h - 8 - 3 };
  }

  private dipY(db: number): number {
    const { y0, y1 } = this.dipBand();
    return y0 + (Math.min(0, Math.max(DIP_DB_MIN, db)) / DIP_DB_MIN) * (y1 - y0);
  }

  /** 谷の底の頂点の位置（谷は子音の頭から広さ w のぶん。頂点はその真ん中） */
  private dipHandle(s: PhoneSeg): { x: number; y: number } {
    const d = s.dip!;
    return { x: this.secToX(s.start + (d.w * (s.end - s.start)) / 2), y: this.dipY(d.db) };
  }

  /** (x, y) の近くの凹みの頂点 */
  private dipAt(x: number, y: number): PhoneSeg | null {
    let best: PhoneSeg | null = null;
    let bd = DIP_HIT_R;
    for (const s of this.derived().phones) {
      if (!s.dip || s.vowel) continue;
      const p = this.dipHandle(s);
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < bd) {
        bd = d;
        best = s;
      }
    }
    return best;
  }

  private xToSec(x: number): number {
    return this.view.xToTick(x) * secPerTick(this.store.project.bpm);
  }

  /** 選んだ頂点（今の音素の並びの中にあるもの） */
  private selected(): PhoneSeg[] {
    return this.derived().phones.filter((s) => s.dip && !s.vowel && this.sel.has(dipKey(s)));
  }

  private inSelRange(x: number): boolean {
    const r = this.selRange;
    if (!r || !this.sel.size) return false;
    const t = this.xToSec(x);
    return t >= r.a && t <= r.b;
  }

  private onDown(e: PointerEvent): void {
    const cv = e.currentTarget as HTMLCanvasElement;
    // 凹みの頂点を先に見る（短い子音では頂点が境界のすぐ近くに来るので。境界は頂点の外ならどの高さでもつかめる）
    const dp = e.button === 0 ? this.dipAt(e.offsetX, e.offsetY) : null;
    if (dp?.dip) {
      const k = dipKey(dp);
      if (e.metaKey || e.ctrlKey) {                    // ⌘ / Ctrl クリック: 選択に足す・外す
        if (this.sel.has(k)) this.sel.delete(k);
        else this.sel.add(k);
        this.redraw();
        return;
      }
      cv.setPointerCapture(e.pointerId);
      this.dipHover = dp;
      if (this.sel.has(k) && this.sel.size > 1) {      // 選んだもののどれか: まとめて上下に動かす
        this.groupDrag = { items: this.selected().map((s) => ({ noteId: s.noteId, idx: s.idx, db0: s.dip!.db })),
                           y: e.offsetY, before: this.store.snapshot() };
        return;
      }
      this.sel = new Set([k]);
      this.selRange = null;
      this.dipDrag = { noteId: dp.noteId, idx: dp.idx, len: dp.end - dp.start, x: e.offsetX, y: e.offsetY,
                       db0: dp.dip.db, w0: dp.dip.w, before: this.store.snapshot() };
      return;
    }
    const s = this.boundaryAt(e.offsetX);
    if (s) {
      cv.setPointerCapture(e.pointerId);
      this.drag = { noteId: s.noteId, idx: s.idx, baseStart: s.baseStart, before: this.store.snapshot() };
      return;
    }
    if (e.button !== 0) return;
    // 空いている所: 動かせば範囲選択、動かさずに離せばクリック（onUp）
    cv.setPointerCapture(e.pointerId);
    this.band = { x0: e.offsetX, x1: e.offsetX, add: e.metaKey || e.ctrlKey, breath: this.breathAt(e.offsetX), moved: false };
  }

  /** ブレスの所のクリック: 入れていなければ入れる、入れてあれば別のパターンに替える */
  private clickBreath(b: BreathSlot): void {
    const n = this.store.track.notes.find((q) => q.id === b.noteId);
    const ctx = this.breathCtx();
    if (!n || !ctx) return;
    const before = this.store.snapshot();
    if (!addOrCycleBreath(n, ctx)) return;
    this.store.commit(before);
    this.store.changed("project");
  }

  /** 右クリック: 入れてあるブレスを消す */
  private onContext(e: MouseEvent): void {
    e.preventDefault();
    const b = this.breathAt(e.offsetX);
    const n = b && b.index !== null ? this.store.track.notes.find((q) => q.id === b.noteId) : undefined;
    if (!n?.breath) return;
    const before = this.store.snapshot();
    delete n.breath;
    this.store.commit(before);
    this.store.changed("project");
  }

  private onMove(e: PointerEvent): void {
    const gd = this.groupDrag;
    if (gd) {
      // 選んだもの全部の深さを同じだけ動かす（広さは変えない。Shift の間は細かく）
      const k = e.shiftKey ? DIP_FINE : 1;
      const { y0, y1 } = this.dipBand();
      const delta = ((e.offsetY - gd.y) * k / (y1 - y0)) * DIP_DB_MIN;
      for (const it of gd.items) {
        const n = this.store.track.notes.find((q) => q.id === it.noteId);
        if (!n) continue;
        const db = Math.min(0, Math.max(DIP_DB_MIN, it.db0 + delta));
        n.phDip = { ...(n.phDip ?? {}), [it.idx]: { ...(n.phDip?.[it.idx] ?? {}), db: Math.round(db * 100) / 100 } };
      }
      this.store.changed("project");
      return;
    }
    const bd = this.band;
    if (bd) {
      bd.x1 = e.offsetX;
      if (Math.abs(bd.x1 - bd.x0) > 3) bd.moved = true;
      this.redraw();
      return;
    }
    const dd = this.dipDrag;
    if (dd) {
      // 動かした量で決める（Shift の間は細かく）。上下 = 深さ、左右 = 広さ（頂点は谷の真ん中なので、広さは動かした秒の 2 倍）
      const n = this.store.track.notes.find((q) => q.id === dd.noteId);
      if (!n) return;
      const k = e.shiftKey ? DIP_FINE : 1;
      const { y0, y1 } = this.dipBand();
      const db = Math.min(0, Math.max(DIP_DB_MIN, dd.db0 + ((e.offsetY - dd.y) * k / (y1 - y0)) * DIP_DB_MIN));
      const spt = secPerTick(this.store.project.bpm);
      const dSec = (this.view.xToTick(e.offsetX) - this.view.xToTick(dd.x)) * spt * k;
      const w = Math.min(1, Math.max(DIP_W_MIN, dd.w0 + (2 * dSec) / Math.max(dd.len, 1e-6)));
      n.phDip = { ...(n.phDip ?? {}), [dd.idx]: { db: Math.round(db * 100) / 100, w: Math.round(w * 1000) / 1000 } };
      this.store.changed("project");
      return;
    }
    const d = this.drag;
    if (!d) {
      const dp = this.dipAt(e.offsetX, e.offsetY);
      if (dp !== this.dipHover) {
        this.dipHover = dp;
        this.redraw();
      }
      const s = dp ? null : this.boundaryAt(e.offsetX);
      const b = s || dp ? null : this.breathAt(e.offsetX);
      this.hc.canvas.style.cursor = dp ? "move" : s ? "ew-resize" : b ? "pointer" : "default";
      const many = dp && this.sel.has(dipKey(dp)) && this.sel.size > 1;
      this.hint(many ? t("ドラッグ: 選んだ子音の凹みの深さをまとめて上下（Shift で細かく）　ダブルクリック: まとめて既定に戻す")
        : dp ? t("ドラッグ: 上下で子音の凹みの深さ、左右で広さ（Shift で細かく）　ダブルクリック: 既定に戻す　⌘クリック: 選択に足す・外す")
        : s ? t("ドラッグ: 音素の境界を移動（休符の後の先頭は前の休符の中まで伸ばせます。フレーズの終わりは動きません）　ダブルクリック: 元に戻す")
        : b ? (b.index === null ? t("クリック: ここにブレスを入れる") : t("クリック: 別のブレスに替える　右クリック: 消す"))
        : this.inSelRange(e.offsetX) ? t("ダブルクリック: 選んだ子音の凹みをまとめて既定に戻す　範囲の外をクリック: 選択を外す")
        : t("横にドラッグ: 範囲の子音の凹みを選ぶ（⌘ を押しながらで足す）"));
      return;
    }
    const n = this.store.track.notes.find((q) => q.id === d.noteId);
    if (!n) return;
    const sec = this.view.xToTick(e.offsetX) * secPerTick(this.store.project.bpm);
    n.phShift = { ...(n.phShift ?? {}), [d.idx]: sec - d.baseStart };
    this.store.changed("project");
  }

  private onUp(): void {
    const bd = this.band;
    if (bd) {
      this.band = null;
      if (bd.moved) {
        // 範囲の中の頂点を選ぶ（⌘ / Ctrl を押していたら今の選択に足す）
        const lo = Math.min(bd.x0, bd.x1);
        const hi = Math.max(bd.x0, bd.x1);
        if (!bd.add) this.sel = new Set();
        for (const s of this.derived().phones) {
          if (!s.dip || s.vowel) continue;
          const x = this.dipHandle(s).x;
          if (x >= lo && x <= hi) this.sel.add(dipKey(s));
        }
        const a = this.xToSec(lo);
        const b = this.xToSec(hi);
        const r = bd.add && this.selRange ? { a: Math.min(a, this.selRange.a), b: Math.max(b, this.selRange.b) } : { a, b };
        this.selRange = this.sel.size ? r : null;
        this.hint(this.sel.size ? t("{n} 個の子音の凹みを選びました（頂点を上下にドラッグでまとめて深さ、範囲の中をダブルクリックで既定に戻す）",
                                    { n: this.sel.size }) : "");
      } else if (this.inSelRange(bd.x0)) {
        // 選んだ範囲の中のクリック: 選択はそのまま（続けてダブルクリックで、まとめて既定に戻すため）
      } else if (bd.breath) {
        this.clickBreath(bd.breath);
      } else {
        this.sel.clear();
        this.selRange = null;
      }
      this.redraw();
      return;
    }
    if (this.drag) this.store.commit(this.drag.before);
    if (this.dipDrag) this.store.commit(this.dipDrag.before);
    if (this.groupDrag) this.store.commit(this.groupDrag.before);
    this.drag = null;
    this.dipDrag = null;
    this.groupDrag = null;
  }

  /** 選んだ凹みを全部、既定に戻す（動かした深さ・広さを消す）。1 回の取り消しで戻る */
  private resetSelected(): void {
    const segs = this.selected().filter((s) => this.store.track.notes.find((q) => q.id === s.noteId)?.phDip?.[s.idx]);
    if (!segs.length) return;
    const before = this.store.snapshot();
    for (const s of segs) {
      const n = this.store.track.notes.find((q) => q.id === s.noteId)!;
      const rest = { ...n.phDip };
      delete rest[s.idx];
      n.phDip = Object.keys(rest).length ? rest : undefined;
    }
    this.store.commit(before);
    this.store.changed("project");
  }

  private onDbl(e: MouseEvent): void {
    // 選んだ頂点か、範囲で選んだ所の中: 選んだものをまとめて既定に戻す
    const dp = this.dipAt(e.offsetX, e.offsetY);
    if (this.sel.size > 1 && (dp ? this.sel.has(dipKey(dp)) : this.inSelRange(e.offsetX))) {
      this.resetSelected();
      return;
    }
    if (!dp && this.inSelRange(e.offsetX)) {
      this.resetSelected();
      return;
    }
    // 凹みの頂点: 動かした深さ・広さを消して既定に戻す
    if (dp) {
      const n = this.store.track.notes.find((q) => q.id === dp.noteId);
      if (!n?.phDip?.[dp.idx]) return;
      const before = this.store.snapshot();
      const rest = { ...n.phDip };
      delete rest[dp.idx];
      n.phDip = Object.keys(rest).length ? rest : undefined;
      this.store.commit(before);
      this.store.changed("project");
      return;
    }
    const s = this.boundaryAt(e.offsetX);
    const n = s && this.store.track.notes.find((q) => q.id === s.noteId);
    if (!s || !n?.phShift?.[s.idx]) return;
    const before = this.store.snapshot();
    const rest = { ...n.phShift };
    delete rest[s.idx];
    n.phShift = Object.keys(rest).length ? rest : undefined;
    this.store.commit(before);
    this.store.changed("project");
  }

  /** 子音の凹み: 上の線（凹みなし）から垂れ下がる谷と、谷の底の頂点。頂点の上・動かしている途中は値も出す */
  private drawDips(ctx: CanvasRenderingContext2D, w: number): void {
    const { y0 } = this.dipBand();
    const accent = cssVar("--accent");
    const soft = cssVar("--accent-soft");
    const hov = this.dipHover;
    const h = this.hc.h;
    // 範囲選択: ドラッグ中の帯と、選んだ範囲（薄い塗りと両端の線）
    const band = this.band?.moved ? [Math.min(this.band.x0, this.band.x1), Math.max(this.band.x0, this.band.x1)] : null;
    const r = !band && this.selRange && this.sel.size ? [this.secToX(this.selRange.a), this.secToX(this.selRange.b)] : null;
    const span = band ?? r;
    if (span) {
      const a = Math.max(KEY_W, span[0]);
      const b = Math.min(w, span[1]);
      if (b > a) {
        ctx.fillStyle = soft;
        ctx.globalAlpha = band ? 0.6 : 0.35;
        ctx.fillRect(a, 0, b - a, h);
        ctx.globalAlpha = 1;
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(Math.round(a) + 0.5, 0);
        ctx.lineTo(Math.round(a) + 0.5, h);
        ctx.moveTo(Math.round(b) - 0.5, 0);
        ctx.lineTo(Math.round(b) - 0.5, h);
        ctx.stroke();
        ctx.restore();
      }
    }
    for (const s of this.derived().phones) {
      const d = s.dip;
      if (!d || s.vowel) continue;
      const x0 = this.secToX(s.start);
      const x1 = this.secToX(s.end);
      if (x1 < KEY_W || x0 > w || x1 - x0 < 3) continue;
      const xe = this.secToX(s.start + d.w * (s.end - s.start));
      // 凹みなしの線（子音の箱の幅だけ）
      ctx.fillStyle = cssVar("--muted");
      ctx.globalAlpha = 0.35;
      ctx.fillRect(x0, Math.round(y0), Math.max(1, x1 - x0 - 1), 1);
      ctx.globalAlpha = 1;
      // 谷（ν 字）: 塗りと線
      const N = 24;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      for (let k = 0; k <= N; k++) ctx.lineTo(x0 + ((xe - x0) * k) / N, this.dipY(dipCurveDb(d.db, k / N)));
      ctx.lineTo(xe, y0);
      ctx.closePath();
      ctx.fillStyle = soft;
      ctx.fill();
      ctx.beginPath();
      for (let k = 0; k <= N; k++) {
        const x = x0 + ((xe - x0) * k) / N;
        const y = this.dipY(dipCurveDb(d.db, k / N));
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // 頂点: 動かしたものは塗り、既定のままのものは輪
      const p = this.dipHandle(s);
      ctx.beginPath();
      ctx.arc(p.x, p.y, DIP_HANDLE_R, 0, 2 * Math.PI);
      ctx.fillStyle = d.edited ? accent : cssVar("--pr-bg");
      ctx.fill();
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1;
      ctx.stroke();
      if (this.sel.has(dipKey(s))) {                     // 選んだ頂点: 外に輪を足す
        ctx.beginPath();
        ctx.arc(p.x, p.y, DIP_HANDLE_R + 2.5, 0, 2 * Math.PI);
        ctx.strokeStyle = cssVar("--text");
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      if (hov && hov.noteId === s.noteId && hov.idx === s.idx) {
        const ms = Math.round(d.w * (s.end - s.start) * 1000);
        const label = `${d.db <= -0.05 ? "−" : ""}${Math.abs(d.db).toFixed(1)} dB · ${ms} ms`;
        ctx.font = "11px 'Hiragino Sans', sans-serif";
        const tw = ctx.measureText(label).width;
        const lx = Math.min(w - tw - 4, Math.max(KEY_W + 2, p.x - tw / 2));
        const ly = p.y + DIP_HANDLE_R + 13 <= this.hc.h - 2 ? p.y + DIP_HANDLE_R + 13 : p.y - DIP_HANDLE_R - 5;
        ctx.fillStyle = cssVar("--panel");
        ctx.globalAlpha = 0.85;
        ctx.fillRect(lx - 3, ly - 11, tw + 6, 15);
        ctx.globalAlpha = 1;
        ctx.fillStyle = cssVar("--text");
        ctx.fillText(label, lx, ly);
        ctx.font = "12px 'Hiragino Sans', sans-serif";
      }
    }
  }

  draw(): void {
    const { ctx, w, h } = this.hc;
    const v = this.view;
    ctx.fillStyle = cssVar("--pr-bg");
    ctx.fillRect(0, 0, w, h);
    // 半透明の箱の下にピアノロールと同じ縦線を通す（位置関係を見やすくする）
    drawTimeGrid(ctx, v, this.store.project.beatsPerBar, this.snapTicks(), 0, h, w);
    ctx.font = "12px 'Hiragino Sans', sans-serif";
    const top = 8;
    const bh = h - 16;
    // ブレス（音素名 br）: 入れてある所はアクセントカラーの塗り、入れられる所は点線の枠と「+ br」（音素の下に描く）
    for (const b of this.derived().breaths) {
      const x0 = this.secToX(b.start + b.cut);
      const x1 = this.secToX(b.end);
      if (x1 < KEY_W || x0 > w) continue;
      const bw = Math.max(2, x1 - x0 - 1);
      const label = b.index === null ? "+ br" : "br";
      if (b.index === null) {
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = cssVar("--accent");
        ctx.globalAlpha = 0.7;
        ctx.lineWidth = 1;
        ctx.strokeRect(Math.round(x0) + 0.5, top + 0.5, Math.round(bw), bh - 1);
        ctx.restore();
      } else {
        ctx.fillStyle = cssVar("--accent-soft");
        ctx.fillRect(x0, top, bw, bh);
      }
      // 名前はほかの音素と同じ位置（左上）に
      const tw = ctx.measureText(label).width;
      if (bw > tw + 4) {
        ctx.fillStyle = b.index === null ? cssVar("--accent") : cssVar("--text");
        ctx.fillText(label, x0 + 5, top + 15);
      }
    }
    for (const seg of this.derived().phones) {
      const x0 = this.secToX(seg.start);
      const x1 = this.secToX(seg.end);
      if (x1 < KEY_W || x0 > w) continue;
      ctx.fillStyle = seg.vowel ? cssVar("--ph-vowel") : cssVar("--ph-cons");
      ctx.fillRect(x0, top, Math.max(1, x1 - x0 - 1), bh);
      ctx.fillStyle = seg.movable ? cssVar("--ph-edge") : cssVar("--ph-edge-fixed");
      ctx.fillRect(Math.round(x0), top - 4, seg.movable ? 1 : 2, bh + 8);
      const tw = ctx.measureText(seg.ph).width;
      if (x1 - x0 > tw + 4) {
        ctx.fillStyle = seg.edited ? cssVar("--ph-edited") : cssVar("--text");
        ctx.fillText(seg.ph, x0 + 5, top + 15);
      }
    }
    this.drawDips(ctx, w);
    const px = v.tickToX(this.store.playTick);
    if (px >= KEY_W) {
      ctx.fillStyle = cssVar("--playhead");
      ctx.fillRect(Math.round(px), 0, 1, h);
    }
    ctx.fillStyle = cssVar("--panel");
    ctx.fillRect(0, 0, KEY_W, h);
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "11px 'Hiragino Sans', sans-serif";
    ctx.fillText(t("音素"), 10, h / 2 + 4);
  }
}

interface PanelDef {
  key: string;
  label: string;
  el: HTMLElement;
}

export interface SizeOpts {
  min: () => number; // 縮められる下限（px）
  max?: number;
  onResize?: (h: number) => void; // 高さを変えたあとに呼ぶ（mel の行の高さを覚えるため）
}

export class BottomPanels {
  private defs: PanelDef[] = [];
  constructor(private container: HTMLElement, private tabbar: HTMLElement, private onLayout: () => void) {}

  add(key: string, label: string, el: HTMLElement, open: boolean, size?: SizeOpts): void {
    el.classList.add("bpanel");
    el.hidden = !open;
    if (size) this.addHandle(el, size);
    this.container.appendChild(el);
    this.defs.push({ key, label, el });
    this.renderTabs();
  }

  /** パネルの上の縁をドラッグして高さを変える（上へ引くと高くなる）。 */
  private addHandle(el: HTMLElement, size: SizeOpts): void {
    const handle = document.createElement("div");
    handle.className = "resize-handle";
    handle.title = t("ドラッグで高さを変える");
    el.appendChild(handle);
    let startY = 0;
    let startH = 0;
    handle.addEventListener("pointerdown", (e) => {
      handle.setPointerCapture(e.pointerId);
      startY = e.clientY;
      startH = el.getBoundingClientRect().height;
      handle.classList.add("dragging");
    });
    handle.addEventListener("pointermove", (e) => {
      if (!handle.classList.contains("dragging")) return;
      const h = Math.min(size.max ?? 600, Math.max(size.min(), startH - (e.clientY - startY)));
      el.style.height = `${Math.round(h)}px`;
      size.onResize?.(h);
    });
    handle.addEventListener("pointerup", () => handle.classList.remove("dragging"));
  }

  private renderTabs(): void {
    this.tabbar.innerHTML = "";
    for (const d of this.defs) {
      const b = document.createElement("button");
      b.className = "tab" + (d.el.hidden ? "" : " on");
      b.innerHTML = `<span>${d.label}</span>`;
      b.addEventListener("click", () => {
        d.el.hidden = !d.el.hidden;
        this.renderTabs();
        this.onLayout();
      });
      this.tabbar.appendChild(b);
    }
  }
}
