// 左の帯の、選んでいるトラック全体の設定: 「歌い手と声」「エフェクター」「レンダリング」を開け閉めできる区切りで並べる。
// 音符ごとの設定は音符の小窓（notepop.ts）、書き出しはダイアログ（dialogs.ts）。
import { addOrCycleBreath } from "./breath";
import type { Derived } from "./derive";
import { singerDisplay } from "./engine";
import { FxPanel } from "./fxpanel";
import { t as tx } from "./i18n";
import { defaultBase, isAudioTrack, type VoiceBase } from "./model";
import { PARAM_GROUPS } from "./params";
import type { Renderer } from "./render";
import type { Store } from "./store";
import { trackColor } from "./theme";

export interface SliderOpts {
  label: string;
  min: number;
  max: number;
  step: number;
  get: () => number;
  set: (v: number) => void;
  fmt: (v: number) => string;
  def: number; // ダブルクリックで戻す値
  disabled?: boolean;
}

/** 名前・つまみ・値の 1 行。動かしている間は store.changed("project")、離したときに undo を積む。 */
export function slider(store: Store, parent: HTMLElement, o: SliderOpts, onDrag: (on: boolean) => void = () => undefined): () => void {
  const row = document.createElement("div");
  row.className = "field" + (o.disabled ? " disabled" : "");
  row.innerHTML = `<span class="f-label"></span><input type="range" min="${o.min}" max="${o.max}" step="${o.step}">
    <span class="val"></span>`;
  row.querySelector(".f-label")!.textContent = o.label;
  const input = row.querySelector("input")!;
  const val = row.querySelector(".val")!;
  input.disabled = !!o.disabled;
  // 中央（既定値）からの塗り（CSS の --lo・--hi）
  const paint = () => {
    const span = o.max - o.min;
    const v = (o.get() - o.min) / span;
    const d = (o.def - o.min) / span;
    input.style.setProperty("--lo", `${Math.min(v, d) * 100}%`);
    input.style.setProperty("--hi", `${Math.max(v, d) * 100}%`);
  };
  const show = () => {
    input.value = String(o.get());
    val.textContent = o.fmt(o.get());
    paint();
  };
  show();
  let before = "";
  input.addEventListener("pointerdown", () => {
    before = store.snapshot();
    onDrag(true);
  });
  input.addEventListener("input", () => {
    if (!before) before = store.snapshot();
    o.set(Number(input.value));
    val.textContent = o.fmt(o.get());
    paint();
    store.changed("project");
  });
  input.addEventListener("change", () => {
    onDrag(false);
    if (before) store.commit(before);
    before = "";
  });
  input.addEventListener("dblclick", () => {
    const b = store.snapshot();
    o.set(o.def);
    store.commit(b);
    show();
    store.changed("project");
  });
  parent.appendChild(row);
  return show;
}

// 声の調整のつまみ: 名前・範囲・刻み・値の書き方・ボイスパラメータか（NSF-HiFiGAN では効かない）。並びと見出しは
// params.ts の PARAM_GROUPS（F0・声質・発音・その他）。F0 の見出しの下なので、抑揚・ビブラートは短い名前にする
const f2 = (v: number): string => v.toFixed(2);
const times = (v: number): string => `× ${v.toFixed(2)}`;
const VOICE_SLIDERS: Partial<Record<keyof VoiceBase, [string, number, number, number, (v: number) => string, boolean]>> = {
  intonationEnv: [tx("抑揚"), 0, 2, 0.01, times, false],
  vibratoEnv: [tx("ビブラート"), 0, 2, 0.01, times, false],
  tension: [tx("テンション"), -1, 1, 0.01, f2, true],
  breathiness: [tx("ブレシネス"), -1, 1, 0.01, f2, true],
  gender: [tx("ジェンダー"), -1, 1, 0.01, f2, true],
  voicing: [tx("有声度"), 0, 1, 0.01, f2, true],
  resonance: [tx("響き"), -1, 1, 0.01, f2, true],
  growl: [tx("うなり"), 0, 1, 0.01, f2, false],
  consonant: [tx("子音の強さ"), -1, 1, 0.01, f2, true],
  articulation: [tx("滑舌"), 0, 2, 0.01, times, false],
  loudness: [tx("音量"), -12, 12, 0.1, (v) => `${v.toFixed(1)} dB`, false],
};

type SecKey = "voice" | "fx" | "render";

export class SidePanel {
  /** エンジン（つながっていれば、レンダリングの状況・モデルの名前・使えるボコーダを出す） */
  engine: Renderer | null = null;
  private open = new Set<SecKey>(["voice", "fx"]);
  private refreshers: Array<() => void> = [];
  private dragging = false;
  private wasConnected = false;
  private fx: FxPanel | null = null; // 開いている「エフェクター」（合成が進んだらイコライザーの後ろのスペクトルを描き直す）

  constructor(private store: Store, private body: HTMLElement, private derived: () => Derived) {
    this.render();
    store.on((kind) => {
      if (kind === "project" || kind === "settings") {
        if (this.dragging) this.refresh();
        else this.render();
      }
    });
  }

  /** 区切りを開く（画面の確認用などで外から）。 */
  openSection(key: SecKey): void {
    this.open.add(key);
    this.render();
  }

  private refresh(): void {
    for (const f of this.refreshers) f();
  }

  /** エンジンの状況が変わったとき。スライダーを動かしている最中は値だけ直す。 */
  update(): void {
    const connected = !!this.engine?.connected;
    this.fx?.refreshSpectrum();
    // 「レンダリング」の見出しの要約は、区切りを閉じていても進み具合に合わせて書き直す
    const rs = this.body.querySelector<HTMLElement>('.acc[data-key="render"] .acc-s');
    if (rs) rs.textContent = this.renderSummary();
    const ae = document.activeElement;
    if (ae && this.body.contains(ae) && (ae.tagName === "INPUT" || ae.tagName === "SELECT")) return;
    if (this.dragging) this.refresh();
    else if (this.open.has("render") || connected !== this.wasConnected) this.render();
    this.wasConnected = connected;
  }

  render(): void {
    const scroll = this.body.scrollTop;
    this.refreshers = [];
    this.fx = null;
    this.body.innerHTML = "";
    const t = this.store.track;
    const audio = isAudioTrack(t);
    if (!audio) {
      this.section("voice", tx("歌い手と声"), `${this.engine?.singerLabel(t) ?? t.voice} · ${this.store.vocoder === "nsf" ? "NSF-HiFiGAN" : "NHVSing"}`,
                   (el) => this.renderVoice(el));
      this.section("fx", tx("エフェクター"), this.fxSummary(), (el) => {
        this.fx = new FxPanel(this.store, (title) => {
          const sec = document.createElement("section");
          sec.className = "side-sec";
          sec.innerHTML = `<h3>${title}</h3>`;
          el.appendChild(sec);
          return sec;
        }, () => (this.engine?.phrases(t) ?? []).flatMap((p) => (p.wav ? [p.wav] : [])));
        this.fx.render();
      });
    }
    this.section("render", tx("レンダリング"), this.renderSummary(), (el) => this.renderStatus(el));
    this.body.scrollTop = scroll;
  }

  /** 開け閉めできる区切り。見出しにはトラックの色の印と要約を出す。 */
  private section(key: SecKey, title: string, summary: string, fill: (el: HTMLElement) => void): void {
    const wrap = document.createElement("div");
    wrap.className = "acc" + (this.open.has(key) ? " open" : "");
    const head = document.createElement("button");
    head.className = "acc-h";
    head.innerHTML = `<span class="acc-car">${this.open.has(key) ? "▾" : "▸"}</span><span class="acc-t"></span><span class="acc-s"></span>`;
    head.querySelector(".acc-t")!.textContent = title;
    head.querySelector(".acc-s")!.textContent = summary;
    wrap.dataset.key = key;
    head.addEventListener("click", () => {
      if (this.open.has(key)) this.open.delete(key);
      else this.open.add(key);
      this.render();
    });
    wrap.appendChild(head);
    if (this.open.has(key)) {
      const body = document.createElement("div");
      body.className = "acc-b";
      wrap.appendChild(body);
      fill(body);
    }
    this.body.appendChild(wrap);
  }

  private fxSummary(): string {
    const fx = this.store.track.fx;
    const n = fx ? [fx.eq.on, fx.comp.on, fx.reverb.on].filter(Boolean).length : 0;
    return n ? tx("{n} つ有効", { n }) : tx("なし");
  }

  private renderSummary(): string {
    const eng = this.engine;
    if (!eng?.connected) return tx("エンジン未接続");
    const ps = eng.phrases(this.store.track);
    const done = ps.filter((p) => p.status === "済み").length;
    return done < ps.length ? tx("{done}/{all} フレーズ", { done, all: ps.length }) : tx("完了");
  }

  private renderVoice(el: HTMLElement): void {
    const s = this.store;
    const t = s.track;
    const head = document.createElement("div");
    head.className = "side-sec";
    // 歌い手の選択肢: 全部のボイスバンクの歌い手（ボイスバンクごとにまとめる）。値は「ボイスバンクの key ⇥ 歌い手の名前」
    const eng = this.engine?.connected ? this.engine : null;
    const cur = eng?.voicebankOf(t);
    const opt = (vb: string, n: string, label: string) =>
      `<option value="${vb}\t${n}" ${vb === (cur?.key ?? "") && n === t.voice ? "selected" : ""}>${label}</option>`;
    const options = eng
      ? eng.info!.voicebanks.map((vb) => eng.info!.voicebanks.length > 1
        ? `<optgroup label="${vb.name}">${vb.singers.map((x) => opt(vb.key, x.name, singerDisplay(x, x.name))).join("")}</optgroup>`
        : vb.singers.map((x) => opt(vb.key, x.name, singerDisplay(x, x.name))).join("")).join("")
      : ["ritsu", "natsume", "oniku"].map((n) => opt("", n, n)).join("");
    const LANG_NAMES: Record<string, string> = { ja: tx("日本語"), ko: tx("韓国語"), en: tx("英語"), zh: tx("中国語") };
    const langCode = cur?.language ?? "ja";
    const lang = LANG_NAMES[langCode] ?? LANG_NAMES[langCode.split("_")[0]] ?? langCode; // en_full などは頭の言語で
    head.innerHTML = `<div class="track-title"><span class="dot"></span><span class="tn"></span></div>
      <div class="row2"><span>${tx("歌い手")}</span><select id="sel-voice">${options}</select></div>
      <div class="row2"><span>${tx("モデル")}</span><span class="muted">${cur ? cur.name
        : tx("エンジン未接続")} · ${lang}</span></div>`;
    head.querySelector<HTMLElement>(".dot")!.style.background = trackColor(s.project, t);
    head.querySelector(".tn")!.textContent = t.name;
    head.querySelector<HTMLSelectElement>("#sel-voice")!.addEventListener("change", (e) => {
      const before = s.snapshot();
      const [vb, name] = (e.target as HTMLSelectElement).value.split("\t");
      t.voice = name;
      if (vb) t.voicebank = vb;
      s.commit(before);
      s.changed("project");
    });
    // ボコーダの切り替えは、エンジンが NSF-HiFiGAN も持つときだけ出す（配布版は NHVSing だけ。Python 版で vocoders/ に置けば出る）
    const nsfOk = !!this.engine?.connected && this.engine.hasVocoder("nsf");
    if (!nsfOk && s.vocoder === "nsf") s.vocoder = "nhv";
    if (nsfOk) {
      const seg = document.createElement("div");
      seg.className = "seg";
      seg.title = tx("ボコーダ");
      for (const [key, label] of [["nhv", "NHVSing v3.2.1"], ["nsf", "NSF-HiFiGAN"]] as const) {
        const b = document.createElement("button");
        b.textContent = label;
        b.className = s.vocoder === key ? "on" : "";
        b.addEventListener("click", () => {
          s.vocoder = key;
          s.changed("settings");
        });
        seg.appendChild(b);
      }
      head.appendChild(seg);
      if (s.vocoder === "nsf") {
        head.insertAdjacentHTML("beforeend", `<p class="hint">${tx("NSF-HiFiGAN は調波・雑音の分岐を持たないため、音量とうなり以外は効きません。")}</p>`);
      }
    }
    el.appendChild(head);

    const base = document.createElement("div");
    base.className = "side-sec";
    base.innerHTML = `<h3>${tx("声の調整")}<span class="h-note">${tx("ダブルクリックで元に戻す")}</span></h3>`;
    // 区分けごとの見出しとつまみ（F0・声質・発音・その他。その他の下にブレスのボタンも置く）
    for (const g of PARAM_GROUPS) {
      const keys = g.keys.filter((k) => k in VOICE_SLIDERS) as Array<keyof VoiceBase>;
      if (!keys.length) continue;
      const head2 = document.createElement("div");
      head2.className = "side-grp";
      head2.textContent = g.label;
      base.appendChild(head2);
      for (const key of keys) {
        const [label, min, max, step, fmt, vp] = VOICE_SLIDERS[key]!;
        this.refreshers.push(slider(s, base, {
          label, min, max, step, fmt, def: defaultBase()[key], disabled: vp && s.vocoder === "nsf",
          get: () => t.base[key], set: (v) => { t.base[key] = v; },
        }, (on) => { this.dragging = on; }));
      }
    }
    // ブレス: 音素のタイミングのパネルで 1 つずつ入れるほか、ここでまとめて入れる・消す。
    // ブレスの束はエンジンから後で届くので（breathContext）、押せるかどうかはエンジンがつながっているかだけで決める
    const br = document.createElement("div");
    br.className = "row2";
    br.innerHTML = `<span>${tx("ブレス")}</span><span><button class="small" id="br-all" ${eng ? "" : "disabled"}>${tx("すべてに入れる")}</button>
      <button class="small" id="br-none" ${eng ? "" : "disabled"}>${tx("すべて消す")}</button></span>`;
    br.querySelector("#br-all")!.addEventListener("click", () => {
      const c = this.engine?.breathContext(t);
      if (!c) return;
      const before = s.snapshot();
      let n = 0;
      for (const slot of this.derived().breaths) {
        if (slot.index !== null) continue;
        const note = t.notes.find((q) => q.id === slot.noteId);
        if (note && addOrCycleBreath(note, c)) n++;
      }
      if (!n) return;
      s.commit(before);
      s.changed("project");
    });
    br.querySelector("#br-none")!.addEventListener("click", () => {
      if (!t.notes.some((q) => q.breath)) return;
      const before = s.snapshot();
      for (const q of t.notes) delete q.breath;
      s.commit(before);
      s.changed("project");
    });
    base.appendChild(br);
    if (eng && eng.breathState(t) === "none") {
      base.insertAdjacentHTML("beforeend", `<p class="hint">${tx("この歌い手にはブレスの素材がありません。")}</p>`);
    } else if (eng) {
      base.insertAdjacentHTML("beforeend", `<p class="hint">${tx("音素のタイミングの「+ br」をクリックしても 1 つずつ入れられます（直前の音符から 0.1 秒以上空いた音符の前）。")}</p>`);
    }
    el.appendChild(base);
  }

  private renderStatus(el: HTMLElement): void {
    const sec = document.createElement("div");
    sec.className = "side-sec";
    el.appendChild(sec);
    const eng = this.engine;
    const st = (s: string) => ({ 解析済み: tx("解析完了"), 合成中: tx("合成中"), 済み: tx("完了"), 失敗: tx("失敗") } as Record<string, string>)[s] ?? s;
    if (eng?.connected) {
      if (eng.error) sec.insertAdjacentHTML("beforeend", `<p class="hint warn">${eng.error}</p>`);
      for (const t of this.store.project.tracks) {
        if (isAudioTrack(t)) continue;
        const ul = document.createElement("ul");
        ul.className = "phrase-list";
        ul.insertAdjacentHTML("beforeend", `<li class="muted"><span></span><span></span><span></span></li>`);
        ul.querySelector("li span")!.textContent = t.name;
        eng.phrases(t).forEach((p, i) => {
          const a = p.startFrame / 172.265625;
          const b = (p.startFrame + p.nFrames) / 172.265625;
          ul.insertAdjacentHTML("beforeend", `<li><span>${tx("フレーズ {i}", { i: i + 1 })}</span><span>${a.toFixed(2)}–${b.toFixed(2)} s</span>
            <span class="${p.status === "済み" ? "ok" : "muted"}">${st(p.status)}</span></li>`);
        });
        sec.appendChild(ul);
      }
      return;
    }
    sec.insertAdjacentHTML("beforeend",
      `<p class="hint">${tx("エンジンは未接続です（モック）。フレーズごとの計算の状況がここに並びます。")}</p>`);
    const d = this.derived();
    const phrases: Array<[number, number]> = [];
    const spans = [...d.noteSec.values()].sort((a, b) => a.start - b.start);
    for (const sp of spans) {
      const last = phrases[phrases.length - 1];
      if (last && sp.start - last[1] < 0.5) last[1] = Math.max(last[1], sp.end);
      else phrases.push([sp.start, sp.end]);
    }
    const ul = document.createElement("ul");
    ul.className = "phrase-list";
    phrases.forEach(([a, b], i) => {
      ul.insertAdjacentHTML("beforeend",
        `<li><span>${tx("フレーズ {i}", { i: i + 1 })}</span><span>${a.toFixed(2)}–${b.toFixed(2)} s</span><span class="muted">${tx("未計算")}</span></li>`);
    });
    sec.appendChild(ul);
  }
}
