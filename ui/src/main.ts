// 組み立て: 状態・表示・各部品をつなぎ、キー操作とツールバーを受け持つ。
import { Arrangement } from "./arrange";
import { BottomPanels, PhonemePanel } from "./bottom";
import { type Derived, derive } from "./derive";
import { openExportDialog, openSettingsDialog } from "./dialogs";
import { buildExports } from "./export";
import { FileMenu } from "./filemenu";
import { localeTag, missingKeys, setLang, t, translateStatic } from "./i18n";
import { NotePopover } from "./notepop";
import { restore } from "./projectfile";
import { parseScoreFile } from "./score";
import { ICONS } from "./icons";
import { defaultFx, isAudioTrack, secPerTick, TPQ, type Track, trackRegion } from "./model";
import { MelPanel } from "./melview";
import { ParamPanel } from "./paramview";
import { PianoRoll, type Tool } from "./pianoroll";
import { Renderer } from "./render";
import { sampleProject } from "./sample";
import { SidePanel } from "./side";
import { Store } from "./store";
import { loadSettings, MEL_MAPS, type MelMapKey, onThemeChange, setAccent, setMelMap, trackColor } from "./theme";
import { Transport } from "./transport";
import { KEY_W, RULER_H, View } from "./view";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
loadSettings(); // 差し色・mel の配色（描く前に CSS の変数を入れる）
translateStatic(); // index.html の文言を今の言語に

const store = new Store(sampleProject());
const view = new View();
view.scrollTick = TPQ * 3;

// エンジン（engine/server.py）。つながらなければ見せかけのデータで動くモックのまま
const renderer = new Renderer(store, () => onEngineUpdate());

let cache: { version: number; track: number; rev: number; d: Derived } | null = null;
const derived = (): Derived => {
  if (!cache || cache.version !== store.version || cache.track !== store.project.activeTrack ||
      cache.rev !== renderer.rev) {
    cache = { version: store.version, track: store.project.activeTrack, rev: renderer.rev,
              d: derive(store.project, store.track, renderer.connected ? renderer.phrases(store.track) : undefined,
                        renderer.breathContext(store.track)) };
  }
  return cache.d;
};
// ブレスの置き場所（再生・書き出しと、書き出す前の確認でトラックごとに要る）。選んでいるトラックは上の取り置きから
renderer.slotsFor = (tr) => (tr.id === store.track.id ? derived().breaths
  : derive(store.project, tr, renderer.connected ? renderer.phrases(tr) : undefined, renderer.breathContext(tr)).breaths);

const status = $("status");
const defaultHint = t("Space: 再生・一時停止　1: 選択　2: 鉛筆　3: ピッチのペン　Delete: 削除　⌘C / ⌘V: コピー / 再生位置に貼り付け　⌘Z / ⇧⌘Z: 取り消し / やり直し　↑↓: 半音（⇧ でオクターブ）");
const hint = (s: string): void => { status.textContent = s || defaultHint; };
hint("");

// ── ピアノロールと編曲 ─────────────────────────
const pr = new PianoRoll(store, view, derived, $("pr-canvas"), $<HTMLInputElement>("lyric-input"), hint);
const arr = new Arrangement(store, view, $("track-list"), $("arr-canvas"),
                            () => Math.max(0, pr.hc.w - KEY_W) / view.pxPerTick, hint, $("btn-add-track"), $("arr-resizer"));
arr.fitHeight();
arr.voiceLabel = (t) => renderer.singerLabel(t); // トラック一覧の歌い手はフルネームで
// 音符の小窓（選んだ音符のそばに出る）
const pop = new NotePopover(store, pr, derived, $("pr-wrap"));
pr.popOpen = () => !pop.el.hidden;
pr.closePop = () => pop.close();

// ── 下のパネル ─────────────────────────────
// タブでパネルを開け閉めしたとき: mel は閉じている間は計算しないので、開いたら計算し直す
const bottom = new BottomPanels($("bottom-panels"), $("tabbar"), () => mel.schedule());
const params = new ParamPanel(store, view, () => pr.snapTicks(), hint, () => derived().phones);
bottom.add("param", t("パラメータ"), params.el, true, { min: () => 32 + 40 });
const phonWrap = document.createElement("div");
phonWrap.style.height = "84px";
const phonCanvas = document.createElement("canvas");
phonWrap.appendChild(phonCanvas);
bottom.add("phoneme", t("音素のタイミング"), phonWrap, true, { min: () => 34, max: 200 });
const phon = new PhonemePanel(store, view, derived, phonCanvas, () => pr.snapTicks(), hint, () => renderer.breathContext(store.track));
const mel = new MelPanel(store, view, derived, () => pr.snapTicks());
bottom.add("mel", "mel", mel.el, true,
           { min: () => mel.minHeight(), max: 900, onResize: (h) => mel.onResize(h) });

// ── 左の帯（選んでいるトラックの設定）・書き出し・設定 ──────────────
const side = new SidePanel(store, $("side-body"), derived);
side.engine = renderer;
$("btn-export").innerHTML = `${ICONS.export}<span>${t("書き出し")}</span>`;
$("btn-export").addEventListener("click", () => openExportDialog(store, renderer));
$("btn-settings").innerHTML = ICONS.settings;
// 言語を変えたら: 待っている自動保存を済ませてから読み込み直す（確認用の ?lang= は外す）
$("btn-settings").addEventListener("click", () => openSettingsDialog(async (l) => {
  setLang(l);
  await files.flushAutosave();
  const u = new URL(location.href);
  u.searchParams.delete("lang");
  location.replace(u.toString());
}));
/** カードの見出し: 今のトラックの色・名前・歌い手。 */
const updateCardHead = (): void => {
  const t = store.track;
  $("cur-dot").style.background = trackColor(store.project, t);
  $("cur-name").textContent = t.name;
  $("cur-voice").textContent = renderer.singerLabel(t);
};
updateCardHead();

// ── 描き直しのつなぎ ───────────────────────────
const timeText = $("time");
const updateTime = (): void => {
  const sec = store.playTick * secPerTick(store.project.bpm);
  const m = Math.floor(sec / 60);
  timeText.textContent = `${String(m).padStart(2, "0")}:${(sec - m * 60).toFixed(2).padStart(5, "0")}`;
};
const redrawAll = (): void => {
  pr.redraw();
  arr.redraw();
  phon.redraw();
  params.redraw();
  mel.redraw();
  pop.update();
};
// 差し色・mel の配色を変えたとき: 色を使うものを全部描き直す
onThemeChange(() => {
  arr.renderList();
  side.render();
  updateCardHead();
  redrawAll();
});
let lastTrack = store.project.activeTrack;
store.on((kind) => {
  if (kind === "project") {
    arr.renderList();
    updateCardHead();
    // テンポと拍子（書き換えた・楽譜を読み込んだ・プロジェクトを開いた・取り消したときに変わる）
    if (!$("bpm-box").querySelector("input") && !$("ts").querySelector("input")) showTempo();
  }
  if (kind === "project" || kind === "settings") {
    mel.schedule();
    renderer.schedule();
  }
  if (store.project.activeTrack !== lastTrack) {
    lastTrack = store.project.activeTrack;
    params.refreshHead();
  }
  if (kind === "playhead") updateTime();
  redrawAll();
});
view.on(redrawAll);
updateTime();

// ── エンジンの状態が変わったとき（解析・合成が 1 句進むたび）────────────
const engineState = document.querySelector<HTMLElement>(".engine-state")!;
const debug = new URLSearchParams(location.search).get("debug") === "1";
let engineHooked = false;
function onEngineUpdate(): void {
  if (renderer.connected && !engineHooked) {
    engineHooked = true;
    // 曲全体の mel の並べ直しは重いので、mel のパネルが版（rev）の変わったときだけ data() を呼ぶ
    mel.setEngine((kind) => ({
      data: () => renderer.trackMel(store.track, kind === "acoustic" ? "in" : kind === "vocoder" ? "plain" : "out"),
      rev: renderer.rev,
    }));
    renderer.wantPlain = () => mel.isOn(1);
    renderer.wantOut = () => mel.isOn(2);
    mel.onToggle = () => renderer.schedule(50);
    transport.engineWav = (t) => renderer.trackWav(t);
    // 再生中に合成し直した句があれば、今の再生位置から鳴らし直す（再生は始めたときの波形を写して鳴らすため）
    renderer.onSynthesized = () => {
      if (transport.playing) void transport.refresh();
    };
    // ピアノロールの下端の波形: 選んでいるトラックの、合成が済んだフレーズとブレスの波形（エフェクトの前）
    pr.waveSource = () => renderer.trackParts(store.track, derived().breaths).map((p) => ({ start: p.start / 44100, wav: p.wav }));
    // 歌い手のフルネームはエンジンの一覧が届いてから分かるので、名前を出す所を描き直す
    side.render();
    arr.renderList();
    updateCardHead();
  }
  document.querySelector(".brand .tag")!.textContent = renderer.connected ? t("実験版") : t("モック");
  if (!renderer.connected) {
    engineState.textContent = t("エンジン未接続（モック）");
  } else {
    const all = store.project.tracks.flatMap((tr) => renderer.phrases(tr));
    const done = all.filter((p) => p.status === "済み").length;
    const pg = renderer.engine.progress; // モデルの読み込み（初回はダウンロード）の進み具合
    const loading = pg ? t("モデルを読み込み中 {p}%（{mb} / {total} MB）", {
      p: Math.floor((100 * pg.loaded) / Math.max(1, pg.total)), mb: (pg.loaded / 1e6).toFixed(0), total: (pg.total / 1e6).toFixed(0) }) : "";
    // 待機中: しばらく操作が無くエンジンが計算の部分を手放している（次の編集で起きる）
    const idle = done < all.length ? t("レンダリング中 {done}/{all} フレーズ", { done, all: all.length })
      : renderer.engine.sleeping ? t("レンダリング完了（待機中）") : t("レンダリング完了");
    engineState.textContent = renderer.error || loading || `${renderer.voicebank.name}　${idle}`;
    engineState.classList.toggle("error", !!renderer.error);
  }
  mel.schedule();
  side.update();
  redrawAll();
  if (debug && renderer.connected) {
    // 確認用: ② と ③ の mel の差（平均の絶対値）と、句ごとの合成の指紋
    const ps = renderer.phrases(store.track);
    const d = ps.map((p) => {
      if (!p.melOut || !p.melPlain) return "-";
      let s = 0;
      for (let i = 0; i < p.melOut.length; i++) s += Math.abs(p.melOut[i] - p.melPlain[i]);
      return (s / p.melOut.length).toFixed(3);
    });
    hint(`debug: rev ${renderer.rev}、②と③の差 [${d.join(", ")}]、指紋 [${ps.map((p) => p.f0Hash?.slice(0, 6)).join(", ")}]`);
  }
}

// ── ツールバー ─────────────────────────────
const tools = $("tools");
const TOOL_DEFS: Array<[Tool, string, string, boolean]> = [
  ["select", "select", t("選択 (1)"), true],
  ["draw", "draw", t("音符を描く (2)"), true],
  ["pitch", "pitchpen", t("ピッチのペン (3)"), true],
];
const renderTools = (): void => {
  tools.innerHTML = "";
  for (const [tool, icon, title, enabled] of TOOL_DEFS) {
    const b = document.createElement("button");
    b.className = "icon-btn" + (pr.tool === tool ? " active" : "");
    b.title = title;
    b.disabled = !enabled;
    b.innerHTML = ICONS[icon];
    b.addEventListener("click", () => setTool(tool));
    tools.appendChild(b);
  }
};
const setTool = (t: Tool): void => {
  pr.tool = t;
  renderTools();
};
renderTools();

const views = $("views");
for (const [key, icon, title] of [["showPitch", "showpitch", t("ピッチの線")], ["showPhonemes", "showphone", t("音素")],
                                  ["showWave", "showwave", t("波形")]] as const) {
  const b = document.createElement("button");
  b.className = "icon-btn on";
  b.title = title;
  b.innerHTML = ICONS[icon];
  b.addEventListener("click", () => {
    pr[key] = !pr[key];
    b.classList.toggle("on", pr[key]);
    pr.redraw();
  });
  views.appendChild(b);
}

$<HTMLSelectElement>("snap").addEventListener("change", (e) => {
  pr.snapSetting = (e.target as HTMLSelectElement).value;
  pr.redraw();
});

// 左の帯の開け閉め（カードの見出しの左端）。開け閉めはこのブラウザに覚えておく（読めなくても開いた状態で動く）
const SIDE_KEY = "svs-editor.sidebar";
const setSide = (open: boolean): void => {
  $("app").classList.toggle("side-closed", !open);
  $("btn-side").classList.toggle("on", !open);
  try {
    localStorage.setItem(SIDE_KEY, open ? "open" : "closed");
  } catch { /* 保存できなくても開け閉めはできる */ }
};
$("btn-side").innerHTML = ICONS.sidebar;
$("btn-side").addEventListener("click", () => setSide($("app").classList.contains("side-closed")));
try {
  if (localStorage.getItem(SIDE_KEY) === "closed") setSide(false);
} catch { /* 読めなければ開いたまま */ }

for (const b of document.querySelectorAll<HTMLButtonElement>(".fold")) {
  b.addEventListener("click", () => {
    const target = $(b.dataset.target!);
    target.classList.toggle("folded");
    b.textContent = target.classList.contains("folded") ? "▸" : "▾";
  });
}

const transport = new Transport(store, view, () => pr.hc.w, () => {
  $("btn-play").innerHTML = transport.playing ? ICONS.pause : ICONS.play;
  $("btn-play").classList.toggle("on", transport.playing);
});
transport.onError = (m) => {
  console.error(m);
  hint(m);
};
$("btn-play").innerHTML = ICONS.play;
$("btn-stop").innerHTML = ICONS.stop;
$("btn-loop").innerHTML = ICONS.loop;
$("btn-follow").innerHTML = ICONS.follow;
$("btn-play").title = t("再生・一時停止（Space）");
$("btn-play").addEventListener("click", () => transport.toggle());
$("btn-stop").addEventListener("click", () => transport.stop());
$("btn-follow").addEventListener("click", () => {
  transport.follow = !transport.follow;
  $("btn-follow").classList.toggle("on", transport.follow);
});
// ── テンポと拍子: 再生の帯の数字を押して書き換える（Enter で決める、Esc でやめる、⌘Z で戻せる）──
const showTempo = (): void => {
  $("bpm").textContent = store.project.bpm.toFixed(2);
  $("ts").textContent = `${store.project.beatsPerBar}/4`;
};
showTempo();
/** テンポを変える。音符・曲線は拍のまま（刻み）。伴奏の音声は秒の位置を保つように範囲を付け直す。 */
const setTempo = (bpm: number): void => {
  const p = store.project;
  if (bpm === p.bpm) return;
  const before = store.snapshot();
  const k = bpm / p.bpm; // 同じ秒の刻みは bpm に比例する
  for (const t of p.tracks) {
    if (!isAudioTrack(t) || !t.region) continue;
    t.region = { start: Math.round(t.region.start * k), end: Math.round(t.region.end * k) };
  }
  p.bpm = bpm;
  store.commit(before);
  store.changed("project");
};
function editReadout(box: HTMLElement, current: () => string, apply: (text: string) => string | null): void {
  box.addEventListener("click", () => {
    if (box.querySelector("input")) return;
    const old = box.innerHTML;
    const inp = document.createElement("input");
    inp.className = "readout-input";
    inp.value = current();
    box.innerHTML = "";
    box.appendChild(inp);
    inp.focus();
    inp.select();
    let done = false;
    const finish = (ok: boolean) => {
      if (done) return;
      done = true;
      const err = ok ? apply(inp.value.trim()) : null;
      box.innerHTML = old;
      showTempo();
      if (err) hint(err);
    };
    inp.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && !e.isComposing) finish(true);
      else if (e.key === "Escape") finish(false);
    });
    inp.addEventListener("blur", () => finish(true));
  });
}
editReadout($("bpm-box"), () => String(store.project.bpm), (s) => {
  const v = Number(s.replace(/[^\d.]/g, ""));
  if (!Number.isFinite(v) || v < 20 || v > 400) return t("テンポは 20〜400 の数で入れてください（{s}）", { s });
  setTempo(Math.round(v * 100) / 100);
  return null;
});
editReadout($("ts"), () => String(store.project.beatsPerBar), (s) => {
  const m = /^(\d+)\s*(?:\/\s*(\d+))?$/.exec(s);
  if (!m || (m[2] && m[2] !== "4")) return t("拍子は「3」「6/4」のように、4 分音符を 1 拍として入れてください（{s}）", { s });
  const n = Number(m[1]);
  if (n < 1 || n > 16) return t("1 小節の拍の数は 1〜16 です");
  if (n !== store.project.beatsPerBar) {
    const before = store.snapshot();
    store.project.beatsPerBar = n;
    store.commit(before);
    store.changed("project");
  }
  return null;
});

// ── キー操作 ───────────────────────────────
window.addEventListener("keydown", (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  const mod = e.metaKey || e.ctrlKey;
  const s = store;
  if (e.key === " ") {
    e.preventDefault();
    transport.toggle();
  } else if (mod && e.key.toLowerCase() === "z") {
    e.preventDefault();
    if (e.shiftKey) s.redo();
    else s.undo();
  } else if (mod && e.key.toLowerCase() === "y") {
    e.preventDefault();
    s.redo();
  } else if (mod && e.key.toLowerCase() === "a") {
    e.preventDefault();
    s.selected = new Set(s.track.notes.map((n) => n.id));
    s.changed("selection");
  } else if (mod && e.key.toLowerCase() === "c") {
    e.preventDefault();
    const n = s.copySelected();
    hint(n ? t("{n} 個の音符をコピーしました", { n }) : t("コピーする音符を選んでください"));
  } else if (mod && e.key.toLowerCase() === "x") {
    e.preventDefault();
    const n = s.cutSelected();
    hint(n ? t("{n} 個の音符を切り取りました", { n }) : t("コピーする音符を選んでください"));
  } else if (mod && e.key.toLowerCase() === "v") {
    e.preventDefault();
    const n = s.paste(); // 今のトラックの再生位置に（別のトラックにも貼れる）
    hint(n ? t("{n} 個の音符を再生位置に貼り付けました", { n }) : t("貼り付ける音符がありません（先に ⌘C でコピー）"));
  } else if (e.key === "Delete" || e.key === "Backspace") {
    if (!s.selected.size) return;
    const before = s.snapshot();
    s.track.notes = s.track.notes.filter((n) => !s.selected.has(n.id));
    s.selected.clear();
    s.commit(before);
    s.changed("project");
    s.changed("selection");
  } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
    if (!s.selected.size) return;
    e.preventDefault();
    const d = (e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 12 : 1);
    const before = s.snapshot();
    for (const n of s.track.notes) if (s.selected.has(n.id)) n.pitch = Math.min(108, Math.max(24, n.pitch + d));
    s.commit(before);
    s.changed("project");
  } else if (e.key === "1") setTool("select");
  else if (e.key === "2") setTool("draw");
  else if (e.key === "3") setTool("pitch");
});

// ── 見本の状態（画面の確認用。?select=3&vocoder=nsf&tab=voice のように使う）────
const q = new URLSearchParams(location.search);
if (q.get("vocoder") === "nsf") {
  store.vocoder = "nsf";
  store.changed("settings");
}
const sel = q.get("select");
if (sel !== null) {
  const sorted = [...store.track.notes].sort((a, b) => a.onset - b.onset);
  const n = sorted[Number(sel)];
  if (n) {
    const vmod = q.get("vmod");
    if (vmod !== null) n.attrs.vMod = Number(vmod);
    const imod = q.get("imod");
    if (imod !== null) n.attrs.iMod = Number(imod);
    store.selected = new Set([n.id]);
    store.changed("project");
    store.changed("selection");
    pop.open();
  }
}
// ── 保存と自動保存 ──────────────────────────────
// 画面の確認用の引数（?demo= など）が付いているときは、自動保存を読み書きしない（使う人の保存を上書きしないように）
// エンジンの選び方（?gpu=1・?engine=server・?idle=）はふだん使いの引数なので、確認用には数えない
const testing = [...q.keys()].some((k) => k !== "gpu" && k !== "engine" && k !== "idle") && !q.has("autosave");
const files = new FileMenu(store, $("btn-file"), $("save-state"), hint, sampleProject, !testing);
// 楽譜を読み込んだら、そのトラックの頭と音域が見えるように編曲とピアノロールを動かす
const revealNotes = (t: Track): void => {
  if (!t.notes.length) return;
  view.scrollTick = Math.max(0, Math.min(...t.notes.map((n) => n.onset)) - TPQ);
  // 音域が収まるなら全体を、収まらなければ低い方から見せる（目盛りの高さを除いた分が見える範囲）。
  // 編曲の欄を広げた直後なので、ピアノロールの高さはその場で測り直す（覚えている値は広げる前のもの）
  const ps = t.notes.map((n) => n.pitch);
  const vis = (pr.hc.canvas.getBoundingClientRect().height - RULER_H) / view.pxPerSemi;
  view.topPitch = Math.min(Math.max(...ps) + 1.5, Math.min(...ps) - 1 + vis);
  view.emit();
};
files.breathContext = (t) => renderer.breathContext(t);
files.onImport = (t) => {
  arr.reveal(t);
  revealNotes(t);
};
// 読み込んだ楽譜の歌詞にハングル（仮名）があり、今の歌い手のボイスバンクが別の言語なら、その言語のボイスバンクの最初の歌い手にする
files.pickVoice = (lyrics, cur) => {
  if (!renderer.connected) return null;
  const has = (re: RegExp) => lyrics.some((l) => re.test(l ?? ""));
  const lang = has(/[가-힣]/) ? "ko" : has(/[぀-ヿ]/) ? "ja" : null;
  if (!lang) return null;
  const vbs = renderer.info!.voicebanks;
  if ((vbs.find((v) => v.key === cur.voicebank) ?? vbs[0]).language === lang) return null;
  const vb = vbs.find((v) => v.language === lang && v.singers.length);
  return vb ? { voicebank: vb.key, voice: vb.singers[0].name } : null;
};
// プロジェクトを開いたら: 曲の地図を曲全体が収まる縮尺・高さにし、今のトラックの頭と音域を見せる
files.onOpen = (t) => {
  arr.fitHeight();
  arr.fitAll();
  revealNotes(t);
};
// 確認用（URL に引数があるときだけ）: 外から楽譜の読み込みなどを呼べるようにする
if (testing) (window as unknown as { __svs: unknown }).__svs = { files, store, view, transport, parseScoreFile, missingKeys, derived, renderer };
if (!testing) {
  void restore().then((r) => {
    if (!r) return;
    store.load(r.project, r.name);
    files.onOpen(store.track);
    hint(t("前回の続きから開きました（{at} に自動保存）", { at: new Date(r.at).toLocaleString(localeTag()) }));
    files.checkMissing(r.project);
  }).catch((e) => hint(t("自動保存を読めませんでした: {e}", { e: String(e) })));
}

void renderer.init(); // エンジンにつなぐ（つながれば自動で描画が始まる）

// 画面の確認用: 差し色・mel の配色を保存せずに変える（?accent=ff8a5c&melmap=ice）、設定のダイアログを開く
const qAccent = q.get("accent");
if (qAccent && /^[0-9a-f]{6}$/i.test(qAccent)) setAccent(`#${qAccent}`, false);
const qMel = q.get("melmap");
if (qMel && qMel in MEL_MAPS) setMelMap(qMel as MelMapKey, false);
if (q.get("demo") === "settings") openSettingsDialog();

const melOn = q.get("mel");
if (melOn) mel.setOn([...melOn].map((c) => c === "1"));
if (q.get("compact") === "1") {
  // 画面の確認用: 下のパネルを一番小さくした状態
  params.el.style.height = "72px";
  phonWrap.style.height = "34px";
  mel.el.style.height = `${mel.minHeight()}px`;
  mel.onResize(mel.minHeight());
}
const later = q.get("later");
if (later) {
  // 画面の確認用: 描画が済んだ後にボイスパラメータを変える（?later=tension:1,gender:1）
  setTimeout(() => {
    for (const kv of later.split(",")) {
      const [k, v] = kv.split(":");
      (store.track.base as unknown as Record<string, number>)[k] = Number(v);
    }
    store.changed("project");
  }, Number(q.get("laterMs") ?? 8000));
}
if (q.get("demo") === "retake") {
  // 画面の確認用: 選んだ音符のピッチを 2 回取り直し、「一つ前」「最初」「一つ前」と押して種の移り変わりを出す
  const log: string[] = [];
  const n = () => store.track.notes.find((x) => store.selected.has(x.id));
  const press = (i: number) => {
    const b = document.querySelectorAll<HTMLButtonElement>(".retake-row")[0].querySelectorAll("button")[i];
    b.click();
    log.push(`${["取り直す", "一つ前", "最初"][i]}→${n()?.attrs.retakeF0 ?? 0}（履歴 ${n()?.attrs.retakeF0Prev?.length ?? 0}）`);
  };
  setTimeout(() => {
    [0, 0, 1, 2, 1].forEach(press);
    hint(`retake: ${log.join(" / ")}`);
  }, 3000);
}
if (q.get("demo") === "curves") {
  // 画面の確認用: 音量とピッチのずれに見本の曲線を入れる
  const t = store.track;
  t.params.loudness = { interp: "cubic", pts: [[1920, 0], [2880, -6], [3840, 3], [4800, -9], [5760, 0]] };
  t.params.pitchDelta = { interp: "cubic", pts: [[2400, 0], [2700, 120], [3000, -80], [3300, 0]] };
  store.changed("project");
}
if (q.get("demo") === "menu") $("btn-add-track").click(); // 画面の確認用: ＋のメニューを開く
if (q.get("demo") === "addvocal") {
  // 画面の確認用: ＋ →「歌のトラック」を押し、トラックの一覧と選ばれたトラックを出す
  $("btn-add-track").click();
  document.querySelector<HTMLButtonElement>(".pop-menu button")?.click();
  hint(`tracks: ${store.project.tracks.map((t) => `${t.name}[${trackRegion(t, 4).start}-${trackRegion(t, 4).end}]`)
    .join(", ")} / active=${store.track.name}`);
}
if (q.get("demo") === "revstat") {
  // 画面の確認用: リバーブの種類ごとの残響の性格（最初の 50 ms の割合・明るさ・−60 dB までの時間）
  void (async () => {
    const F = await import("./fx");
    const ctx = new OfflineAudioContext(2, 1, 44100);
    const out: string[] = [];
    for (const [k, p] of Object.entries(F.REVERB_PRESETS)) {
      const x = F.reverbIR(ctx, k as keyof typeof F.REVERB_PRESETS, p.decay).getChannelData(0);
      const sr = 44100;
      let total = 0;
      let early = 0;
      for (let i = 0; i < x.length; i++) {
        total += x[i] * x[i];
        if (i < 0.05 * sr) early += x[i] * x[i];
      }
      let zc = 0;
      const a = Math.floor(0.1 * sr);
      const b = Math.floor(0.3 * sr);
      for (let i = a + 1; i < b; i++) if ((x[i - 1] < 0) !== (x[i] < 0)) zc++;
      // 残りのエネルギー（後ろから足した量）が −60 dB になる時間
      let acc = 0;
      let t60 = 0;
      for (let i = x.length - 1; i >= 0; i--) {
        acc += x[i] * x[i];
        if (acc > total * 1e-6 && !t60) t60 = i / sr;
      }
      out.push(`${p.label}: 最初の50ms ${(100 * early / total).toFixed(0)}% 明るさ ${Math.round((zc / ((b - a) / sr)) / 2)}Hz 長さ ${t60.toFixed(2)}s`);
    }
    hint(`revstat: ${out.join(" / ")}`);
  })();
}
if (q.get("fx")) {
  // 画面の確認用: メロディにエフェクトの見本の値を入れ、エフェクトのパネルを開く
  // ?fx=1 で全部、?fx=eq・comp・rev で 1 つだけ
  const which = q.get("fx")!;
  const fx = defaultFx();
  fx.eq.on = which === "1" || which === "eq";
  fx.eq.lowCut = 120;
  fx.eq.bands = [{ freq: 300, gain: -3, q: 1.2 }, { freq: 1500, gain: 6, q: 1 }, { freq: 8000, gain: -4, q: 0.8 }];
  fx.comp.on = which === "1" || which === "comp";
  fx.reverb.on = which === "1" || which === "rev";
  fx.reverb.wet = 0.35;
  store.track.fx = fx;
  side.openSection("fx");
  store.changed("mix");
}
if (q.get("demo") === "export") {
  // 画面の確認用: 描画が済んだら全部（ミックス＋各トラック）を作って中身を調べる
  store.exportOpts.trackIds = store.project.tracks.map((t) => t.id);
  store.exportOpts.bitDepth = Number(q.get("bits") ?? 16) as 16 | 24 | 32;
  store.exportOpts.channels = Number(q.get("ch") ?? 2) as 1 | 2;
  const wait = () => {
    if (!renderer.connected || renderer.pending() > 0) {
      setTimeout(wait, 300);
      return;
    }
    // ※ ヘッドレスの Chrome の仮想時間（--virtual-time-budget）の下では OfflineAudioContext が進まないので、実時間で確かめる
    const all = (["mix", "tracks"] as const).map((w) => buildExports(store.project, (t) => renderer.trackWav(t), store.exportOpts, w));
    void Promise.all(all).then((r) => r.flat()).then(async (files) => {
      const info = await Promise.all(files.map(async (f) => {
        const v = new DataView(await f.blob.arrayBuffer());
        const ch = v.getUint16(22, true);
        const bits = v.getUint16(34, true);
        const n = v.getUint32(40, true) / (ch * bits / 8);
        let ss = 0;
        for (let i = 0; i < n; i += 7) {
          const o = 44 + i * ch * bits / 8;
          const x = bits === 16 ? v.getInt16(o, true) / 32768 : bits === 32 ? v.getFloat32(o, true)
            : ((v.getUint8(o) | (v.getUint8(o + 1) << 8) | (v.getInt8(o + 2) << 16)) / 8388608);
          ss += x * x;
        }
        const rms = 10 * Math.log10(ss / Math.ceil(n / 7) + 1e-12);
        return `${f.name} ${ch}ch ${bits}bit ${v.getUint32(24, true)}Hz ${(n / 44100).toFixed(2)}s ${rms.toFixed(1)}dB`;
      }));
      hint(`export: ${info.join(" | ")}`);
    });
  };
  setTimeout(wait, 1000);
}
if (q.get("demo") === "arrtest") {
  // 画面の確認用: 編曲を拡大・右へスクロールし、2 つ目のトラックがあれば右クリックのメニューで消し、メロディの名前を変える
  setTimeout(() => {
    const c = $("arr-canvas");
    c.dispatchEvent(new WheelEvent("wheel", { deltaY: -110, ctrlKey: true, clientX: 0, bubbles: true, cancelable: true }));
    c.dispatchEvent(new WheelEvent("wheel", { deltaX: 120, bubbles: true, cancelable: true }));
    const rows = document.querySelectorAll<HTMLElement>(".track-row");
    rows[1]?.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 200, clientY: 200 }));
    document.querySelectorAll<HTMLButtonElement>(".pop-menu button")[1]?.click();
    document.querySelector<HTMLElement>(".track-row .name")?.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    const inp = document.querySelector<HTMLInputElement>(".track-row input.rename");
    if (inp) {
      inp.value = "主旋律";
      inp.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    }
    hint(`arrtest: ${store.project.tracks.map((t) => t.name).join(", ")} / active=${store.track.name}`);
  }, 2500);
}
/** 画面の確認用: 8 秒の和音（拍ごとに弱くなる）の wav。 */
function demoWav(): File {
  const sr = 22050;
  const n = sr * 8;
  const pcm = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const env = Math.exp(-((t * 80 / 60) % 1) * 3);
    const v = [220, 277.2, 329.6].reduce((acc, f) => acc + Math.sin(2 * Math.PI * f * t), 0) / 3;
    pcm[i] = Math.round(v * env * 0.6 * 32767);
  }
  const head = new DataView(new ArrayBuffer(44));
  const str = (o: number, s: string) => { for (let k = 0; k < 4; k++) head.setUint8(o + k, s.charCodeAt(k)); };
  str(0, "RIFF"); head.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt ");
  head.setUint32(16, 16, true); head.setUint16(20, 1, true); head.setUint16(22, 1, true);
  head.setUint32(24, sr, true); head.setUint32(28, sr * 2, true); head.setUint16(32, 2, true);
  head.setUint16(34, 16, true); str(36, "data"); head.setUint32(40, n * 2, true);
  return new File([head.buffer, pcm.buffer], "伴奏の見本.wav", { type: "audio/wav" });
}
if (q.get("demo") === "audio" || q.get("demo") === "arrtest") {
  // 画面の確認用: 伴奏のトラックとして 2 小節目に足す
  void arr.addAudioTrack(demoWav(), store.project.beatsPerBar * TPQ);
}
if (q.get("demo") === "savetest") {
  // 画面の確認用: 伴奏を足し、JSON・zip・自動保存の 3 通りで保存して読み戻し、元と同じかを調べる
  void (async () => {
    await arr.addAudioTrack(demoWav(), store.project.beatsPerBar * TPQ);
    const P = await import("./projectfile");
    const A = await import("./audiofiles");
    const orig = JSON.stringify(store.project);
    const out: string[] = [];
    const same = (p: unknown) => JSON.stringify(p) === orig;
    out.push(`json=${same(P.migrate(P.serialize(store.project)))}`);
    const zipBlob = P.bundle(store.project);
    const z = await P.openBundle(await zipBlob.arrayBuffer());
    const aid = z.tracks.find((t) => t.audio)?.audio?.id ?? "";
    out.push(`zip=${same(z)} ${(zipBlob.size / 1e3).toFixed(0)}KB audio=${!!A.audioEntry(aid)}`);
    await P.autosave(store.project, "保存の確認");
    const r = await P.restore();
    out.push(`autosave=${!!r && same(r.project)} name=${r?.name}`);
    // 伴奏が手元に無いときのつなぎ直し: 指紋で照合する
    const p2 = P.migrate(P.serialize(store.project));
    for (const t of p2.tracks) if (t.audio) t.audio.id = "missing1";
    out.push(`missing=${P.missingAudio(p2).length}`);
    const rl = await P.relink(p2, [demoWav()]);
    out.push(`relink=${rl.linked.join(",")} left=${P.missingAudio(p2).length}`);
    await P.clearAutosave();
    hint(`savetest: ${out.join(" / ")}`);
  })();
}
if (q.get("mix")) {
  // 画面の確認用: つまみに見本の値を入れる
  const [a, b] = store.project.tracks;
  Object.assign(a, { gain: -6.5, pan: -0.35 });
  Object.assign(b, { gain: 2, pan: 0.6 });
  arr.renderList();
}
if (q.get("shift")) {
  // 画面の確認用: メロディのトラックを中身ごと動かす（箱のドラッグと同じ）
  const dt = Number(q.get("shift"));
  const t = store.track;
  const r = trackRegion(t, store.project.beatsPerBar);
  t.region = { start: r.start + dt, end: r.end + dt };
  for (const n of t.notes) n.onset += dt;
  for (const c of Object.values(t.params)) if (c) c.pts = c.pts.map(([tk, v]) => [tk + dt, v]);
  store.changed("project");
}
if (q.get("phshift")) {
  // 画面の確認用: ?phshift=2:1:0.15 → 3 つ目の音符の 2 つ目の音素の頭を 0.15 秒後ろへ
  const [ni, pi, sec] = q.get("phshift")!.split(":").map(Number);
  const n = [...store.track.notes].sort((a, b) => a.onset - b.onset)[ni];
  if (n) n.phShift = { [pi]: sec };
  store.changed("project");
}
if (q.get("trim")) {
  // 画面の確認用: メロディの範囲を 3 つ目の音符（ら）の途中で終える
  const t = store.track;
  const n = [...t.notes].sort((a, b) => a.onset - b.onset)[2];
  t.region = { start: trackRegion(t, store.project.beatsPerBar).start, end: n.onset + n.dur / 2 };
  store.changed("project");
}
