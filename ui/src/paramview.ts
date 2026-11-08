// パラメータのパネル: 調声の曲線をタブで並べ、重ねて見せて、いちばん手前の曲線を鉛筆・直線・消しゴムで描く。
//   タブ: [可視化] [名前（押すと手前に出す）] [⇕（パラメータを選び直す）] [×]、並びの右に [＋]（曲線を足す）
//   重なり: 最近押した順に手前。2 枚目は少し薄く、3 枚目からだんだん薄く。描くのはいちばん手前の曲線
//   見せる値は「曲線 ＋ トラック全体の値」（有声度は掛け算。engine.ts の phraseParams と同じ）を範囲の端で止めたもの。
//   左の帯でトラック全体の値を動かすと曲線全体が上下に動く。描いたときは、見えている値になるよう曲線の値を逆算して書く
// 半透明の地にピアノロールと同じ縦線を通して、時間の位置を揃えて見せる。
import { cssVar, HiDpiCanvas, requestDraw } from "./canvas";
import type { PhoneSeg } from "./derive";
import { drawTimeGrid } from "./grid";
import { t as tr } from "./i18n";
import { ICONS } from "./icons";
import { FPS, type ParamKey, secPerTick, type Track, type VoiceBase } from "./model";
import { EraseStroke, ensureCurve, evalCurve, lineRange, MULTIPLY_KEYS, PARAMS, paramDef, PenStroke } from "./params";
import type { Store } from "./store";
import { KEY_W, type View, wheelScrollX } from "./view";

type PTool = "pen" | "line" | "eraser";
const PAD = 10;
const PH_GAP = 5; // 音素の境目の線・塗りの、上下の隙間（px）

/** 曲線の色（重ねたときに見分けられるよう、パラメータごとに決める）。 */
const PARAM_COLOR: Record<ParamKey, string> = {
  pitchDelta: "#ffa3cd", intonationEnv: "#c6e86a", vibratoEnv: "#f5c542", loudness: "#7fd0c6", tension: "#ff8a5c", breathiness: "#9fc0d6",
  gender: "#b38cff", voicing: "#5b8cff", consonant: "#f0896b", resonance: "#e3e6ff", growl: "#ff5d73",
  articulation: "#7ee0a1",
};
/** 手前から k 枚目（0 = いちばん手前）の濃さ。 */
const layerAlpha = (k: number): number => (k === 0 ? 1 : k === 1 ? 0.6 : Math.max(0.12, 0.6 * 0.6 ** (k - 1)));

const EYE_ON = '<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z"/><circle cx="8" cy="8" r="2" fill="currentColor" stroke="none"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z" opacity="0.5"/><path d="M2.5 13.5l11-11"/></svg>';
const UPDOWN = '<svg viewBox="0 0 10 16" width="9" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 6l3-3 3 3M2 10l3 3 3-3"/></svg>';

/** トラック全体の値（左の帯の「声の調整」）。ピッチのずれ・ビブラートの強さには無い。 */
function baseOf(track: Track, key: ParamKey): number | null {
  return key in track.base ? track.base[key as keyof VoiceBase] : null;
}

export class ParamPanel {
  readonly el: HTMLElement;
  readonly hc: HiDpiCanvas;
  private tabs: ParamKey[] = ["pitchDelta"]; // タブの並び（足した順）
  private order: ParamKey[] = ["pitchDelta"]; // 重なりの順（[0] がいちばん手前 = 描く曲線）
  private hidden = new Set<ParamKey>(); // 可視化を切った曲線
  private tool: PTool = "pen";
  // 鉛筆・消しゴムは一筆の範囲だけを書き換える（params.ts の PenStroke・EraseStroke。外側は描き始めの前のまま）
  private drag: { kind: "pen" | "erase" | "line"; t: number; v: number; t1: number; v1: number; before: string;
                  pen?: PenStroke; eraser?: EraseStroke } | null = null;
  private head: HTMLElement;

  /** phones: 音素の区間（音素の境目を薄く重ねるため。derive の phones） */
  constructor(private store: Store, private view: View, private snapTicks: () => number,
              private hint: (s: string) => void, private phones: () => PhoneSeg[] = () => []) {
    this.el = document.createElement("div");
    this.el.className = "param-panel";
    this.el.style.height = "150px";
    this.head = document.createElement("div");
    this.head.className = "pp-head";
    const canvas = document.createElement("canvas");
    this.el.append(this.head, canvas);
    this.hc = new HiDpiCanvas(canvas, () => this.draw());
    canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", (e) => this.onUp(e));
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("wheel", (e) => wheelScrollX(this.view, e), { passive: false }); // 横スクロールと横の拡大縮小
    this.renderHead();
  }

  /** 描く曲線（いちばん手前）。 */
  get editKey(): ParamKey {
    return this.order[0];
  }

  private bringToFront(key: ParamKey): void {
    this.order = [key, ...this.order.filter((k) => k !== key)];
    this.hidden.delete(key); // 手前に出したものは見せる
  }

  private renderHead(): void {
    const used = new Set(this.tabs);
    const tabHtml = this.tabs.map((key, i) => {
      const d = paramDef(key);
      const opts = PARAMS.filter((p) => p.key === key || !used.has(p.key))
        .map((p) => `<option value="${p.key}" ${p.key === key ? "selected" : ""}>${p.label}</option>`).join("");
      const vis = !this.hidden.has(key);
      return `<div class="pp-tab${key === this.editKey ? " front" : ""}" data-i="${i}" style="--pc:${PARAM_COLOR[key]}">
        <button class="pp-eye${vis ? " on" : ""}" title="${tr("表示/非表示")}">${vis ? EYE_ON : EYE_OFF}</button>
        <button class="pp-name" title="${tr("クリックで前面に表示（編集できるのは最前面の曲線）")}">${d.label}</button>
        <label class="pp-pick" title="${tr("パラメータを選び直す")}">${UPDOWN}<select>${opts}</select></label>
        ${this.tabs.length > 1 ? `<button class="pp-close" title="${tr("この曲線を閉じる")}">✕</button>` : ""}
      </div>`;
    }).join("");
    this.head.innerHTML = `<div class="pp-tabs">${tabHtml}
        ${this.tabs.length < PARAMS.length ? `<button class="icon-btn pp-add" title="${tr("曲線を追加")}">＋</button>` : ""}</div>
      <div class="tool-group">
        <button class="icon-btn ${this.tool === "pen" ? "active" : ""}" data-tool="pen" title="${tr("鉛筆")}">${ICONS.draw}</button>
        <button class="icon-btn ${this.tool === "line" ? "active" : ""}" data-tool="line" title="${tr("直線")}">${ICONS.line}</button>
        <button class="icon-btn ${this.tool === "eraser" ? "active" : ""}" data-tool="eraser" title="${tr("消しゴム（右ドラッグでも消せる）")}">${ICONS.eraser}</button>
      </div>
      <span class="pp-value muted"></span>`;
    const refresh = () => {
      this.renderHead();
      this.redraw();
    };
    for (const tab of this.head.querySelectorAll<HTMLElement>(".pp-tab")) {
      const key = this.tabs[Number(tab.dataset.i)];
      tab.querySelector(".pp-name")!.addEventListener("click", () => {
        this.bringToFront(key);
        refresh();
      });
      tab.querySelector(".pp-eye")!.addEventListener("click", () => {
        if (this.hidden.has(key)) this.hidden.delete(key);
        else this.hidden.add(key);
        refresh();
      });
      tab.querySelector("select")!.addEventListener("change", (e) => {
        // タブのパラメータを選び直す（重なりの位置・見せるかどうかはそのまま引き継ぐ）
        const nk = (e.target as HTMLSelectElement).value as ParamKey;
        const swap = (k: ParamKey) => (k === key ? nk : k);
        this.tabs = this.tabs.map(swap);
        this.order = this.order.map(swap);
        if (this.hidden.delete(key)) this.hidden.add(nk);
        refresh();
      });
      tab.querySelector(".pp-close")?.addEventListener("click", () => {
        this.tabs = this.tabs.filter((k) => k !== key);
        this.order = this.order.filter((k) => k !== key);
        this.hidden.delete(key);
        refresh();
      });
    }
    this.head.querySelector(".pp-add")?.addEventListener("click", () => {
      const next = PARAMS.find((p) => !this.tabs.includes(p.key));
      if (!next) return;
      this.tabs.push(next.key);
      this.order.push(next.key);
      this.bringToFront(next.key);
      refresh();
    });
    for (const b of this.head.querySelectorAll<HTMLButtonElement>("[data-tool]")) {
      b.addEventListener("click", () => {
        this.tool = b.dataset.tool as PTool;
        this.renderHead();
      });
    }
  }

  /** トラックを切り替えたときなどに見出しを作り直す。 */
  refreshHead(): void {
    this.renderHead();
  }

  redraw(): void {
    requestDraw(() => this.draw());
  }

  // ── 値（曲線とトラック全体の値を合わせる）───────────────────
  /** 見せる値: 曲線の値とトラック全体の値を合わせて、範囲の端で止める。 */
  private shown(key: ParamKey, tick: number): number {
    const d = paramDef(key);
    const cv = evalCurve(this.store.track.params[key], tick, d.def);
    const b = baseOf(this.store.track, key);
    const v = b === null ? cv : MULTIPLY_KEYS.has(key) ? b * cv : b + cv;
    return Math.min(d.max, Math.max(d.min, v));
  }

  /** 描いた（見えている）値から、曲線に書く値を逆算する。 */
  private toCurve(key: ParamKey, v: number): number {
    const b = baseOf(this.store.track, key);
    if (b === null) return v;
    if (MULTIPLY_KEYS.has(key)) return b > 1e-3 ? v / b : v;
    return v - b;
  }

  // ── 座標 ─────────────────────────────────
  private yOf(key: ParamKey, v: number): number {
    const d = paramDef(key);
    return PAD + ((d.viewMax - v) / (d.viewMax - d.viewMin)) * (this.hc.h - 2 * PAD);
  }
  /** y の位置の見えている値（範囲の端で止める）。 */
  private vOf(key: ParamKey, y: number): number {
    const d = paramDef(key);
    return Math.min(d.max, Math.max(d.min, d.viewMax - ((y - PAD) / (this.hc.h - 2 * PAD)) * (d.viewMax - d.viewMin)));
  }

  // ── 操作 ─────────────────────────────────
  /** エンジンの 1 フレームの長さ（刻み）。 */
  private frameTicks(): number {
    return 1 / FPS / secPerTick(this.store.project.bpm);
  }
  /** 鉛筆が点を置く間隔（刻み）: 画面の 3 px。ただし縮小しても 2 フレームより粗くしない（速い動きを取りこぼさない）。 */
  private penGap(): number {
    return Math.min(3 / this.view.pxPerTick, 2 * this.frameTicks());
  }
  /** 一筆の最小の幅（刻み）: 3 フレーム。これより細い変化はエンジンでならされて聞こえない。 */
  private minStroke(): number {
    return 3 * this.frameTicks();
  }

  private onDown(e: PointerEvent): void {
    if (e.offsetX < KEY_W) return;
    (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
    const key = this.editKey;
    const t = this.view.xToTick(e.offsetX);
    const v = this.vOf(key, e.offsetY);
    const kind = e.button === 2 || this.tool === "eraser" ? "erase" : this.tool === "line" ? "line" : "pen";
    this.drag = { kind, t, v, t1: t, v1: v, before: this.store.snapshot() };
    this.hidden.delete(key); // 描くなら見せる
    const c = ensureCurve(this.store.track.params, key);
    const def = paramDef(key).def;
    if (kind === "pen") {
      this.drag.pen = new PenStroke(c, def, this.penGap());
      this.drag.pen.add(t, this.toCurve(key, v), false);
      this.store.changed("project");
    } else if (kind === "erase") {
      this.drag.eraser = new EraseStroke(c, def);
      this.drag.eraser.add(t);
      this.store.changed("project");
    }
  }

  private onMove(e: PointerEvent): void {
    const key = this.editKey;
    const t = this.view.xToTick(Math.max(KEY_W, e.offsetX));
    const v = this.vOf(key, e.offsetY);
    const d = paramDef(key);
    this.head.querySelector(".pp-value")!.textContent = `${d.label}: ${d.fmt(this.shown(key, t))}`;
    const g = this.drag;
    if (!g) {
      this.hint(tr("ドラッグ: 最前面の曲線を描く　右ドラッグ: 消去（初期値に戻す）　タブ名をクリックで前面に表示"));
      return;
    }
    if (g.pen) {
      // ブラウザが 1 回にまとめた途中の動きも使う（速く動かしたときの山や谷を取りこぼさない）
      const evs = typeof e.getCoalescedEvents === "function" ? e.getCoalescedEvents() : [];
      for (const ev of evs.length ? evs : [e]) {
        g.pen.add(this.view.xToTick(Math.max(KEY_W, ev.offsetX)), this.toCurve(key, this.vOf(key, ev.offsetY)));
      }
      this.store.changed("project");
    } else if (g.eraser) {
      g.eraser.add(t);
      this.store.changed("project");
    } else {
      g.t1 = t;
      g.v1 = v;
      this.redraw();
    }
  }

  private onUp(e: PointerEvent): void {
    const g = this.drag;
    this.drag = null;
    if (!g) return;
    const key = this.editKey;
    if (g.pen) {
      // 指を離した位置も描き、細すぎる一筆は 3 フレームの幅にする
      g.pen.add(this.view.xToTick(Math.max(KEY_W, e.offsetX)), this.toCurve(key, this.vOf(key, e.offsetY)));
      g.pen.finish(this.minStroke());
      this.store.changed("project");
    }
    if (g.kind === "line" && (g.t1 !== g.t || g.v1 !== g.v)) {
      // 縦に引いた直線（幅がほぼ無いもの）も、終わりの値を 3 フレームの幅で置く
      const c = ensureCurve(this.store.track.params, key);
      lineRange(c, g.t, this.toCurve(key, g.v), g.t1, this.toCurve(key, g.v1), this.penGap(), paramDef(key).def,
                this.minStroke());
      this.store.changed("project");
    }
    const c = this.store.track.params[key];
    if (c && c.pts.length === 0) delete this.store.track.params[key];
    this.store.commit(g.before);
    this.renderHead();
    this.redraw();
  }

  // ── 描画 ─────────────────────────────────
  /**
   * 音素の境目（名前は出さない）。子音などの区間を薄く塗り、核音素（母音・N）の頭は点線、それ以外の音素の頭と
   * 休符に入る所は少し濃い実線。端から端まで通る拍の線と見分けるため、上下に PH_GAP の隙間を空ける（下の音素の欄と同じ）。
   * pau の中には描かない。
   */
  private drawPhones(): void {
    const segs = this.phones();
    if (!segs.length) return;
    const { ctx, w, h } = this.hc;
    const spt = secPerTick(this.store.project.bpm);
    const xs = (sec: number) => this.view.tickToX(sec / spt);
    const y0 = PH_GAP;
    const y1 = h - PH_GAP;
    const accent = cssVar("--accent");
    ctx.fillStyle = accent;
    ctx.globalAlpha = 0.09;
    for (const s of segs) {
      if (s.vowel) continue;
      const a = Math.max(KEY_W, xs(s.start));
      const b = Math.min(w, xs(s.end));
      if (b > a) ctx.fillRect(a, y0, b - a, y1 - y0);
    }
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1;
    const line = (x: number, dotted: boolean) => {
      if (x < KEY_W || x > w) return;
      const xx = Math.round(x) + 0.5;
      ctx.globalAlpha = dotted ? 0.4 : 0.55;
      ctx.setLineDash(dotted ? [2, 3] : []);
      ctx.beginPath();
      ctx.moveTo(xx, y0);
      ctx.lineTo(xx, y1);
      ctx.stroke();
    };
    segs.forEach((s, i) => {
      line(xs(s.start), s.vowel);
      const next = segs[i + 1];
      if (!next || next.start - s.end > 1e-4) line(xs(s.end), false); // 休符（pau）に入る所
    });
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  draw(): void {
    const { ctx, w, h } = this.hc;
    const params = this.store.track.params;
    ctx.fillStyle = cssVar("--pr-bg");
    ctx.fillRect(0, 0, w, h);
    drawTimeGrid(ctx, this.view, this.store.project.beatsPerBar, this.snapTicks(), 0, h, w);
    this.drawPhones();

    const front = this.editKey;
    const ed = paramDef(front);
    // トラック全体の値の線（曲線が既定値のときの位置。左の帯で動かすとここも動く）
    const baseY = Math.round(this.yOf(front, this.shown(front, -1e9)));
    ctx.fillStyle = cssVar("--accent");
    ctx.globalAlpha = 0.5;
    ctx.fillRect(KEY_W, baseY, w - KEY_W, 1);
    ctx.globalAlpha = 1;

    // 奥から手前へ重ねる（見せる曲線だけ）
    const layers = this.order.filter((k) => !this.hidden.has(k));
    for (let k = layers.length - 1; k >= 0; k--) {
      const key = layers[k];
      ctx.globalAlpha = layerAlpha(k);
      ctx.strokeStyle = PARAM_COLOR[key];
      ctx.lineWidth = k === 0 ? 1.8 : 1.3;
      ctx.beginPath();
      for (let x = KEY_W; x <= w; x += 2) {
        const y = this.yOf(key, this.shown(key, this.view.xToTick(x)));
        if (x === KEY_W) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // いちばん手前の曲線の点
    if (!this.hidden.has(front)) {
      const pts = params[front]?.pts ?? [];
      const visible = pts.filter(([t]) => this.view.tickToX(t) >= KEY_W && this.view.tickToX(t) <= w);
      if (visible.length && visible.length < (w - KEY_W) / 8) {
        ctx.fillStyle = PARAM_COLOR[front];
        for (const [t] of visible) {
          ctx.beginPath();
          ctx.arc(this.view.tickToX(t), this.yOf(front, this.shown(front, t)), 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    // 直線の道具の下書き
    if (this.drag?.kind === "line") {
      const g = this.drag;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = PARAM_COLOR[front];
      ctx.beginPath();
      ctx.moveTo(this.view.tickToX(g.t), this.yOf(front, g.v));
      ctx.lineTo(this.view.tickToX(g.t1), this.yOf(front, g.v1));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // NSF-HiFiGAN では効かないパラメータ
    if (ed.voiceParam && this.store.vocoder === "nsf") {
      ctx.fillStyle = cssVar("--outside");
      ctx.fillRect(KEY_W, 0, w - KEY_W, h);
      ctx.fillStyle = cssVar("--muted");
      ctx.font = "12px 'Hiragino Sans', sans-serif";
      ctx.fillText(tr("NSF-HiFiGAN では効きません"), KEY_W + 12, h / 2 + 4);
    }
    // 再生位置
    const px = this.view.tickToX(this.store.playTick);
    if (px >= KEY_W) {
      ctx.fillStyle = cssVar("--playhead");
      ctx.fillRect(Math.round(px), 0, 1, h);
    }
    // 左の欄: いちばん手前の曲線の縦軸の両端の言葉
    ctx.fillStyle = cssVar("--panel");
    ctx.fillRect(0, 0, KEY_W, h);
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "11px 'Hiragino Sans', sans-serif";
    ctx.fillText(ed.top, 8, Math.min(PAD + 8, h / 2 - 2));
    if (h >= 56) ctx.fillText(ed.bottom, 8, h - PAD); // 縮めたときは上端の言葉だけ
  }
}
