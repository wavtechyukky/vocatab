// ダイアログ: 書き出し（WAV）と設定（言語・差し色・mel の配色・英語の歌詞の + の読み方）。
import { buildExports, download, exportFileName, type ExportWhat } from "./export";
import { getLang, type Lang, LANGS, t } from "./i18n";
import type { LyricStyle } from "./model";
import type { Renderer } from "./render";
import type { Store } from "./store";
import {
  ACCENT_PRESETS, DEFAULT_ACCENT, getAccent, getGpu, getMelMap, MEL_MAPS, type MelMapKey, melGradient, setAccent, setGpu,
  setMelMap, trackColor,
} from "./theme";

/** 英語の歌詞の + の読み方: ボタンの文字と説明 */
const LYRIC_STYLES: LyricStyle[] = ["sequential", "count"];
const lyricStyleLabel = (k: LyricStyle): string => (k === "count" ? t("+ の数で音節を指定") : t("+ は次の音節へ"));
const lyricStyleHint = (k: LyricStyle): string => (k === "count"
  ? t("+ の数だけ音節を歌います（++ で 2 音節）。余った音節は歌いません。")
  : t("+ で次の音節へ進みます（++ も 1 音節）。音節が余れば最後の音符でまとめて歌います。"));

/** 画面の真ん中に出す枠。外側を押すか Esc で閉じる。 */
function modal(title: string, onClose: () => void = () => undefined): { box: HTMLElement; close: () => void } {
  const back = document.createElement("div");
  back.className = "modal-back";
  const box = document.createElement("div");
  box.className = "modal dialog";
  box.innerHTML = `<div class="dlg-head"><h3></h3><button class="icon-btn" title="${t("閉じる（Esc）")}">✕</button></div>`;
  box.querySelector("h3")!.textContent = title;
  back.appendChild(box);
  document.body.appendChild(back);
  const close = () => {
    back.remove();
    window.removeEventListener("keydown", onKey, true);
    onClose();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      close();
    }
  };
  window.addEventListener("keydown", onKey, true);
  back.addEventListener("pointerdown", (e) => { if (e.target === back) close(); });
  box.querySelector(".dlg-head button")!.addEventListener("click", close);
  return { box, close };
}

/** 書き出し: ミックス 1 本が主役（大きなボタン）。トラックを別々に出したいときは下を開いて、行のスイッチで選ぶ（計算は export.ts）。
 *  音はいつも再生と同じ（ミュート・ソロ、トラックの音量とパン、エフェクターを反映）。 */
export function openExportDialog(store: Store, engine: Renderer | null): void {
  const { box } = modal(t("書き出し"), () => clearInterval(timer));
  const o = store.exportOpts;
  const p = store.project;
  const body = document.createElement("div");
  body.className = "dlg-body";
  body.innerHTML = `
    <div class="row2"><span>${t("ファイル名")}</span><input id="ex-name" type="text"></div>
    <div class="row2"><span>${t("形式")}</span><div class="pair">
      <select id="ex-ch" title="${t("チャンネル")}"><option value="2">${t("ステレオ")}</option><option value="1">${t("モノ（パンは掛けない）")}</option></select>
      <select id="ex-bits" title="${t("ビット深度")}"><option value="16">16 bit</option><option value="24">24 bit</option>
        <option value="32">${t("32 bit 浮動小数")}</option></select>
    </div></div>
    <button id="ex-mix" class="wide-btn primary"></button>
    <p class="hint" id="ex-mix-hint"></p>
    <details class="ex-more" id="ex-more">
      <summary><span>${t("トラックを別々に書き出す")}</span><span class="muted" id="ex-count"></span></summary>
      <div id="ex-list"></div>
      <button id="ex-tracks" class="wide-btn"></button>
    </details>
    <p class="hint" id="ex-msg"></p>`;
  box.appendChild(body);
  const $ = <T extends HTMLElement>(id: string) => body.querySelector<T>(`#${id}`)!;
  const name = $<HTMLInputElement>("ex-name");
  name.value = o.fileName;
  name.addEventListener("input", () => { o.fileName = name.value.trim() || t("無題"); renderList(); update(); });
  const ch = $<HTMLSelectElement>("ex-ch");
  ch.value = String(o.channels);
  ch.addEventListener("change", () => { o.channels = Number(ch.value) as 1 | 2; });
  const bits = $<HTMLSelectElement>("ex-bits");
  bits.value = String(o.bitDepth);
  bits.addEventListener("change", () => { o.bitDepth = Number(bits.value) as 16 | 24 | 32; });

  // トラックを別々に: 行（色・名前・できるファイル名・スイッチ）を押すと入れ外し。前に選んでいたら開いた状態で出す
  const list = $("ex-list");
  const more = $<HTMLDetailsElement>("ex-more");
  o.trackIds = o.trackIds.filter((id) => p.tracks.some((tr) => tr.id === id));
  more.open = o.trackIds.length > 0;
  const renderList = () => {
    list.innerHTML = "";
    for (const tr of p.tracks) {
      const on = o.trackIds.includes(tr.id);
      const row = document.createElement("div");
      row.className = "ex-track" + (on ? "" : " off");
      row.innerHTML = `<i class="dot"></i><span class="name"></span><span class="file"></span>
        <button class="switch${on ? " on" : ""}" role="switch" aria-checked="${on}"></button>`;
      row.querySelector<HTMLElement>(".dot")!.style.background = trackColor(p, tr);
      const nm = row.querySelector<HTMLElement>(".name")!;
      nm.textContent = tr.name;
      nm.title = tr.name;
      const file = row.querySelector<HTMLElement>(".file")!;
      file.textContent = on ? exportFileName(o.fileName, tr) : "—";
      file.title = file.textContent;
      row.addEventListener("click", () => {
        o.trackIds = on ? o.trackIds.filter((x) => x !== tr.id) : [...o.trackIds, tr.id];
        renderList();
        update();
      });
      list.appendChild(row);
    }
    $("ex-count").textContent = `${o.trackIds.length} / ${p.tracks.length}`;
  };

  const mixBtn = $<HTMLButtonElement>("ex-mix");
  const trkBtn = $<HTMLButtonElement>("ex-tracks");
  const mixHint = $("ex-mix-hint");
  const msg = $("ex-msg");
  let exporting: ExportWhat | null = null;
  const update = () => {
    const pending = engine?.connected ? engine.pending() : -1;
    const ready = pending === 0 && exporting === null;
    const n = o.trackIds.length;
    mixBtn.disabled = !ready;
    trkBtn.disabled = !ready || n === 0;
    mixBtn.textContent = exporting === "mix" ? t("書き出し中…") : t("ミックスを書き出す");
    trkBtn.textContent = exporting === "tracks" ? t("書き出し中…") : t("選んだトラックを書き出す（{n} ファイル）", { n });
    mixHint.textContent = `${exportFileName(o.fileName)} · ${t("再生と同じ音（ミュート・ソロ、トラックの音量とパン、エフェクターを反映）")}`;
    if (!msg.dataset.done) {
      msg.textContent = pending < 0 ? t("エンジンにつながっていないので書き出せません。")
        : pending > 0 ? t("レンダリング中（残り {n} フレーズ）", { n: pending }) : "";
    }
  };
  renderList();
  update();
  // レンダリングが進むと書き出せるようになるので、開いている間は時々見直す
  const timer = window.setInterval(update, 500);
  const run = (what: ExportWhat) => {
    if (!engine) return;
    exporting = what;
    delete msg.dataset.done;
    update();
    void buildExports(p, (tr) => engine.trackWav(tr), o, what).then((files) => {
      download(files);
      msg.dataset.done = "1";
      msg.textContent = t("書き出しました: {names}", { names: files.map((f) => f.name).join(", ") });
    }).catch((e) => {
      msg.dataset.done = "1";
      msg.textContent = t("書き出しに失敗しました: {e}", { e: String(e) });
    }).finally(() => {
      exporting = null;
      update();
    });
  };
  mixBtn.addEventListener("click", () => run("mix"));
  trkBtn.addEventListener("click", () => run("tracks"));
}

/** 設定: 言語・差し色（音符・強調の色）・mel の配色・エンジン（GPU）。ブラウザに保存する。
 *  英語の歌詞の + の読み方だけはプロジェクトに保存する（store があるとき）。
 *  onLang: 言語を選んだとき（画面を読み込み直す。自動保存を済ませてから） */
export function openSettingsDialog(onLang: (l: Lang) => void = () => undefined, store?: Store): void {
  const { box } = modal(t("設定"));
  const body = document.createElement("div");
  body.className = "dlg-body";
  box.appendChild(body);
  const render = () => {
    const acc = getAccent();
    const mm = getMelMap();
    const ls: LyricStyle = store?.project.lyricStyle ?? "sequential";
    body.innerHTML = `
      <div class="dlg-sub">${t("画面の言語")}<span class="h-note">${t("変更するとページを再読み込みします（編集内容は自動保存から復元されます）")}</span></div>
      <div class="seg langs">${LANGS.map(([k, label]) => `<button data-lang="${k}" class="${k === getLang() ? "on" : ""}">${label}</button>`).join("")}</div>
      <div class="dlg-sub">${t("アクセントカラー")}<span class="h-note">${t("音符・選んだ所・つまみの色。トラックの色もここから作ります")}</span></div>
      <div class="swatches">${ACCENT_PRESETS.map((c) => `<button class="sw${c === acc ? " on" : ""}" data-c="${c}" style="background:${c}" title="${c}"></button>`).join("")}
        <label class="sw-pick" title="${t("好きな色を選ぶ")}"><input type="color" id="acc-pick" value="${acc}"><span>${t("自由選択")}</span></label>
        <button class="small" id="acc-reset">${t("既定に戻す")}</button></div>
      <div class="dlg-sub">${t("mel の配色")}<span class="h-note">${t("下のパネルの mel の色")}</span></div>
      <div class="melmaps">${(Object.keys(MEL_MAPS) as MelMapKey[]).map((k) => `
        <button class="melmap${k === mm ? " on" : ""}" data-k="${k}"><span class="mm-bar" style="background:${melGradient(k)}"></span>
          <span class="mm-name">${MEL_MAPS[k].label}</span></button>`).join("")}</div>
      ${store ? `
      <div class="dlg-sub">${t("英語の歌詞の + 記号")}<span class="h-note">${t("このプロジェクトに保存します")}</span></div>
      <div class="seg lyric-style">${LYRIC_STYLES.map((k) => `<button data-ls="${k}" class="${k === ls ? "on" : ""}">${lyricStyleLabel(k)}</button>`).join("")}</div>
      <p class="hint">${lyricStyleHint(ls)} ${t("どちらも - と +~ は伸ばし、+3 は単語の 3 番目の音素から始めます。")}</p>` : ""}
      <div class="dlg-sub">${t("エンジン")}</div>
      <label class="check" id="gpu-label"><input id="gpu-on" type="checkbox"> ${t("GPU で計算する（実験的・Chrome 向け。変更は読み込み直しで反映）")}</label>
      <div class="check-note" id="gpu-note" hidden></div>`;
    for (const b of body.querySelectorAll<HTMLButtonElement>("[data-lang]")) {
      b.addEventListener("click", () => {
        if (b.dataset.lang !== getLang()) onLang(b.dataset.lang as Lang);
      });
    }
    for (const b of body.querySelectorAll<HTMLButtonElement>(".sw")) {
      b.addEventListener("click", () => { setAccent(b.dataset.c!); render(); });
    }
    const pick = body.querySelector<HTMLInputElement>("#acc-pick")!;
    pick.addEventListener("input", () => setAccent(pick.value));
    pick.addEventListener("change", () => render());
    body.querySelector("#acc-reset")!.addEventListener("click", () => { setAccent(DEFAULT_ACCENT); render(); });
    for (const b of body.querySelectorAll<HTMLButtonElement>(".melmap")) {
      b.addEventListener("click", () => { setMelMap(b.dataset.k as MelMapKey); render(); });
    }
    for (const b of body.querySelectorAll<HTMLButtonElement>("[data-ls]")) {
      b.addEventListener("click", () => {
        if (!store || b.dataset.ls === ls) return;
        const before = store.snapshot();
        if (b.dataset.ls === "count") store.project.lyricStyle = "count";
        else delete store.project.lyricStyle; // 既定（sequential）は書かない
        store.commit(before);
        store.changed("project");
        render();
      });
    }
    // GPU（WebGPU）は既定でオフ。Safari では WebGPU 入りの計算の部品を読むだけで読み込み直しが起きるので、Chrome 向けの実験的な設定
    const gpu = body.querySelector<HTMLInputElement>("#gpu-on")!;
    gpu.checked = getGpu();
    gpu.addEventListener("change", () => setGpu(gpu.checked));
    // 使えない環境では押せなくし（灰色）、理由を下に出す。エンジンも同じ条件で GPU を使わない（worker.ts の loadOrt）
    void gpuUnavailable().then((why) => {
      if (!why || !gpu.isConnected) return;
      gpu.disabled = true;
      gpu.checked = false;
      body.querySelector("#gpu-label")!.classList.add("disabled");
      const note = body.querySelector<HTMLElement>("#gpu-note")!;
      note.textContent = why;
      note.hidden = false;
    });
  };
  render();
}

/** GPU（WebGPU）がこのブラウザで使えなければ、その理由。使えれば null */
async function gpuUnavailable(): Promise<string | null> {
  const ua = navigator.userAgent;
  if (/AppleWebKit/.test(ua) && !/Chrome|Chromium|Edg\//.test(ua)) return t("Safari では使えません（WebGPU 入りの部品を読むとページが読み込み直されるため）");
  const g = (navigator as any).gpu;
  if (!g) return t("このブラウザは WebGPU に対応していません");
  try {
    if (await g.requestAdapter()) return null;
  } catch {
    // 下へ
  }
  return t("この環境では GPU を使えません（WebGPU の準備ができません）");
}
