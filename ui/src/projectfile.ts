// プロジェクトの保存と読み込み。
//   プロジェクトのファイル（.svsproj、JSON）: 伴奏の音声は参照（ファイル名・長さ・指紋）だけ。開くときにつなぎ直す
//   音声も含めた書き出し（.zip）: project.svsproj と audio/ の音声を 1 つにまとめる（共有・移動用）
//   自動保存（ブラウザの IndexedDB）: プロジェクトと伴奏の音声。読み込み直しても続きから
// 形式の版（FORMAT_VERSION）を書いておき、読むときに migrate で今の形に直す。項目を足したら版を上げてここに直し方を書く。
import { audioEntry, registerAudio } from "./audiofiles";
import { t } from "./i18n";
import { defaultBase, isAudioTrack, NOTE_ATTR_KEYS, type NoteAttrs, type Project, trackRegion } from "./model";

export const FORMAT = "svs-editor-project";
// 版 2: 子音の凹み（音符の phDip）・滑舌（声の設定 base.articulation と曲線 params.articulation）・抑揚とビブラートの強さの
// トラック全体の値（base.intonationEnv・base.vibratoEnv。曲線と掛け合わせる）・英語の + の読み方（lyricStyle）を足した。
// 版 1 のファイルはそのまま読める（無い値は既定値: 凹みは既定、滑舌 0、抑揚・ビブラートの強さ 1、+ は次の音節へ）。
// 版を上げたので、前の版のアプリは版 2 のファイルを「新しいバージョン」として開かない（足した値を黙って落とさない）
export const FORMAT_VERSION = 2;

export interface ProjectFile {
  format: string;
  formatVersion: number;
  savedAt: string;
  project: Project;
}

export function serialize(p: Project): string {
  const f: ProjectFile = { format: FORMAT, formatVersion: FORMAT_VERSION, savedAt: new Date().toISOString(), project: p };
  return JSON.stringify(f, null, 1);
}

/** 読んだ JSON を今の形のプロジェクトにする。新しすぎる版や形式違いは理由を付けて投げる。 */
export function migrate(text: string): Project {
  let f: Partial<ProjectFile>;
  try {
    f = JSON.parse(text) as Partial<ProjectFile>;
  } catch {
    throw new Error(t("プロジェクトのファイルとして読めません（JSON ではない）"));
  }
  if (f.format !== FORMAT || !f.project) throw new Error(t("この SVS のプロジェクトのファイルではありません"));
  const v = f.formatVersion ?? 0;
  if (v > FORMAT_VERSION) throw new Error(t("新しいバージョン（{v}）で保存されたファイルです。アプリを更新してください", { v }));
  const p = f.project;
  // 版 1 → 2: 直すものは無い（足した値は無ければ既定値。下の埋め方で済む）
  // 版 1・2: 決めていない値は既定値で埋める（古いファイルや手で書いたファイルのため）
  p.bpm ||= 120;
  p.beatsPerBar ||= 4;
  if (p.lyricStyle !== "count") delete p.lyricStyle; // 英語の + の読み方: 既定（sequential）は書かない
  p.tracks ??= [];
  for (const t of p.tracks) {
    t.notes ??= [];
    // 音符の設定は今の項目だけ残す（以前のモックのピッチの線用の遷移・ビブラートの値は使わないので落とす）
    for (const n of t.notes) {
      const a = (n.attrs ?? {}) as Partial<NoteAttrs>;
      const kept: Record<string, unknown> = { vMod: a.vMod ?? 1 };
      for (const k of NOTE_ATTR_KEYS) if (a[k] !== undefined) kept[k] = a[k];
      n.attrs = kept as unknown as NoteAttrs;
    }
    t.params ??= {};
    // 声の設定: 後から足した項目（滑舌など）が無い古いファイルは既定値で埋める
    t.base = { ...defaultBase(), ...(t.base ?? {}) };
    t.mute = !!t.mute;
    t.solo = !!t.solo;
    if (!t.region) t.region = trackRegion(t, p.beatsPerBar);
  }
  if (!p.tracks.some((x) => !isAudioTrack(x))) throw new Error(t("歌のトラックがありません"));
  if (!p.tracks[p.activeTrack] || isAudioTrack(p.tracks[p.activeTrack])) {
    p.activeTrack = p.tracks.findIndex((t) => !isAudioTrack(t));
  }
  return p;
}

/** 伴奏のうち、まだ読み込まれていない（音声が手元に無い）もの。 */
export function missingAudio(p: Project): Array<{ id: string; name: string; fileName?: string; hash?: string }> {
  const out: Array<{ id: string; name: string; fileName?: string; hash?: string }> = [];
  for (const t of p.tracks) {
    const a = t.audio;
    if (isAudioTrack(t) && a && !audioEntry(a.id) && !out.some((x) => x.id === a.id)) {
      out.push({ id: a.id, name: a.name, fileName: a.fileName, hash: a.hash });
    }
  }
  return out;
}

/** 選んでもらったファイルを、足りない伴奏につなぐ（指紋が同じものを優先、無ければ同じファイル名）。 */
export async function relink(p: Project, files: File[]): Promise<{ linked: string[]; unmatched: string[] }> {
  const miss = missingAudio(p);
  const linked: string[] = [];
  const used = new Set<File>();
  const loaded: Array<{ file: File; bytes: ArrayBuffer; hash: string }> = [];
  for (const f of files) {
    const bytes = await f.arrayBuffer();
    const d = await crypto.subtle.digest("SHA-256", bytes);
    loaded.push({ file: f, bytes, hash: [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("") });
  }
  for (const m of miss) {
    const hit = loaded.find((l) => !used.has(l.file) && m.hash && l.hash === m.hash)
      ?? loaded.find((l) => !used.has(l.file) && m.fileName && l.file.name === m.fileName);
    if (!hit) continue;
    used.add(hit.file);
    await registerAudio(hit.bytes, hit.file.name, m.id, hit.hash);
    linked.push(m.name + (m.hash && hit.hash !== m.hash ? t("（中身が保存したときと違います）") : ""));
  }
  return { linked, unmatched: loaded.filter((l) => !used.has(l.file)).map((l) => l.file.name) };
}

// ── zip（無圧縮）。音声はすでに圧縮されていることが多いので、圧縮しない ──
const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(b: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export function zip(files: Array<{ name: string; data: Uint8Array }>): Blob {
  const enc = new TextEncoder();
  const parts: BlobPart[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  for (const f of files) {
    const name = enc.encode(f.name);
    const crc = crc32(f.data);
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); // UTF-8 の名前
    h.setUint16(8, 0, true); h.setUint32(14, crc, true); h.setUint32(18, f.data.length, true);
    h.setUint32(22, f.data.length, true); h.setUint16(26, name.length, true);
    parts.push(h.buffer, name, f.data);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
    c.setUint32(16, crc, true); c.setUint32(20, f.data.length, true); c.setUint32(24, f.data.length, true);
    c.setUint16(28, name.length, true); c.setUint32(42, offset, true);
    const ce = new Uint8Array(46 + name.length);
    ce.set(new Uint8Array(c.buffer), 0);
    ce.set(name, 46);
    central.push(ce);
    offset += 30 + name.length + f.data.length;
  }
  const size = central.reduce((a, c) => a + c.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
  end.setUint32(12, size, true); end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end.buffer], { type: "application/zip" });
}

/** このアプリで書いた（無圧縮の）zip を読む。 */
export function unzip(buf: ArrayBuffer): Map<string, Uint8Array> {
  const v = new DataView(buf);
  const dec = new TextDecoder();
  const out = new Map<string, Uint8Array>();
  let e = buf.byteLength - 22;
  while (e >= 0 && v.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) throw new Error(t("zip として読めません"));
  const n = v.getUint16(e + 10, true);
  let p = v.getUint32(e + 16, true);
  for (let i = 0; i < n; i++) {
    if (v.getUint32(p, true) !== 0x02014b50) throw new Error(t("zip の目次が壊れています"));
    const method = v.getUint16(p + 10, true);
    const csize = v.getUint32(p + 20, true);
    const nlen = v.getUint16(p + 28, true);
    const xlen = v.getUint16(p + 30, true);
    const clen = v.getUint16(p + 32, true);
    const local = v.getUint32(p + 42, true);
    const name = dec.decode(new Uint8Array(buf, p + 46, nlen));
    if (method !== 0) throw new Error(t("圧縮された zip には対応していません（{name}）", { name }));
    const lstart = local + 30 + v.getUint16(local + 26, true) + v.getUint16(local + 28, true);
    out.set(name, new Uint8Array(buf, lstart, csize));
    p += 46 + nlen + xlen + clen;
  }
  return out;
}

/** 音声も含めた書き出し: project.svsproj と audio/<id>_<元の名前>。 */
export function bundle(p: Project): Blob {
  const enc = new TextEncoder();
  const files = [{ name: "project.svsproj", data: enc.encode(serialize(p)) }];
  const seen = new Set<string>();
  for (const t of p.tracks) {
    const e = t.audio && audioEntry(t.audio.id);
    if (!e || seen.has(e.id)) continue;
    seen.add(e.id);
    files.push({ name: `audio/${e.id}_${e.name}`, data: new Uint8Array(e.bytes) });
  }
  return zip(files);
}

/** zip を開く: プロジェクトと、中の音声を登録する。 */
export async function openBundle(buf: ArrayBuffer): Promise<Project> {
  const files = unzip(buf);
  const pj = files.get("project.svsproj");
  if (!pj) throw new Error(t("zip の中に project.svsproj がありません"));
  const p = migrate(new TextDecoder().decode(pj));
  for (const [name, data] of files) {
    const m = /^audio\/([^_]+)_(.+)$/.exec(name);
    if (m) await registerAudio(data.slice().buffer, m[2], m[1]);
  }
  return p;
}

// ── 自動保存（IndexedDB）──────────────────────────
const DB = "svs-editor";

function db(): Promise<IDBDatabase> {
  return new Promise((ok, ng) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => {
      r.result.createObjectStore("autosave");
      r.result.createObjectStore("audio");
    };
    r.onsuccess = () => ok(r.result);
    r.onerror = () => ng(r.error);
  });
}

function put(d: IDBDatabase, store: string, key: string, value: unknown): Promise<void> {
  return new Promise((ok, ng) => {
    const tx = d.transaction(store, "readwrite");
    tx.objectStore(store).put(value, key);
    tx.oncomplete = () => ok();
    tx.onerror = () => ng(tx.error);
  });
}

function get<T>(d: IDBDatabase, store: string, key: string): Promise<T | undefined> {
  return new Promise((ok, ng) => {
    const r = d.transaction(store).objectStore(store).get(key);
    r.onsuccess = () => ok(r.result as T | undefined);
    r.onerror = () => ng(r.error);
  });
}

/** プロジェクトと、まだ入れていない伴奏の音声をブラウザの中に保存する。 */
export async function autosave(p: Project, name: string): Promise<void> {
  const d = await db();
  for (const t of p.tracks) {
    const e = t.audio && audioEntry(t.audio.id);
    if (e && !(await get(d, "audio", e.id))) await put(d, "audio", e.id, { name: e.name, hash: e.hash, bytes: e.bytes });
  }
  await put(d, "autosave", "current", { text: serialize(p), name, at: Date.now() });
  d.close();
}

/** 自動保存したプロジェクトを読む（伴奏の音声も登録する）。無ければ null。 */
export async function restore(): Promise<{ project: Project; name: string; at: number } | null> {
  const d = await db();
  const rec = await get<{ text: string; name: string; at: number }>(d, "autosave", "current");
  if (!rec) {
    d.close();
    return null;
  }
  const p = migrate(rec.text);
  for (const t of p.tracks) {
    if (!t.audio || audioEntry(t.audio.id)) continue;
    const a = await get<{ name: string; hash: string; bytes: ArrayBuffer }>(d, "audio", t.audio.id);
    if (a) await registerAudio(a.bytes, a.name, t.audio.id, a.hash);
  }
  d.close();
  return { project: p, name: rec.name, at: rec.at };
}

/** 自動保存を消す（新規のとき）。 */
export async function clearAutosave(): Promise<void> {
  const d = await db();
  await new Promise<void>((ok) => {
    const tx = d.transaction(["autosave", "audio"], "readwrite");
    tx.objectStore("autosave").clear();
    tx.objectStore("audio").clear();
    tx.oncomplete = () => ok();
    tx.onerror = () => ok();
  });
  d.close();
}
