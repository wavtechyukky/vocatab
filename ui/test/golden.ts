// 照らし合わせの道具（node で動かす）: Python のエンジンが書き出した正解（tools/golden/dump_golden.py）を読み、配列を比べる。
//   配列は型付きの配列になる（float32 → Float32Array、float64 → Float64Array、int32・int64 → Int32Array、bool → Uint8Array、
//   float16 → Float32Array、文字列 → string[]）。形は shapeOf(配列) で引ける。
// 走らせ方（svs_editor_browser の頭で）:
//   npx esbuild@0.25.0 ui/test/<名前>.test.ts --bundle --platform=node --format=esm --outfile=<作業用>/t.mjs
//   node <作業用>/t.mjs
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const shapes = new WeakMap<object, number[]>();
export const shapeOf = (a: object): number[] => shapes.get(a) ?? [];

/**
 * リポジトリの頭のフォルダ。環境変数 REPO_ROOT があればそれ。無ければカレントディレクトリが頭でも、その 1 つ上でもよい。
 * esbuild でまとめると import.meta.url は出力先になるので、カレントディレクトリから探す
 */
export function repoRoot(): string {
  const cwd = process.cwd();
  const cands = [process.env.REPO_ROOT, cwd, join(cwd, "svs_editor_browser"), join(cwd, ".."),
                 join(dirname(fileURLToPath(import.meta.url)), "../..")].filter((c): c is string => !!c);
  for (const c of cands) {
    try {
      readFileSync(join(c, "web_index.json"));
      return c;
    } catch { /* 次を試す */ }
  }
  throw new Error("svs_editor_browser のフォルダが見つかりません（その頭か research で走らせる）");
}

/** svs_editor_browser の頭から見た相対パスのファイルを読む（web.json など） */
export const readRepo = (rel: string): string => readFileSync(join(repoRoot(), rel), "utf-8");

/** 正解のあるフォルダ。既定は ui/test/golden（環境変数 GOLDEN で変えられる） */
function goldenDir(): string {
  return process.env.GOLDEN ?? join(repoRoot(), "ui/test/golden");
}

function resolve(v: any, dir: string): any {
  if (Array.isArray(v)) return v.map((x) => resolve(x, dir));
  if (v && typeof v === "object") {
    if ("$dtype" in v) {
      const dt = v.$dtype as string;
      const shape = v.$shape as number[];
      if (dt === "str") return v.$data;
      let arr: any;
      if (v.$bin) {
        const b = readFileSync(join(dir, v.$bin));
        const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
        arr = dt === "float64" ? new Float64Array(buf) : dt === "int32" || dt === "int64" ? new Int32Array(buf)
          : dt === "uint8" || dt === "bool" ? new Uint8Array(buf) : new Float32Array(buf);
      } else {
        const d = (v.$data as Array<number | null>).map((x) => x ?? NaN); // null は NaN（JSON に書けないので null にしてある）
        arr = dt === "float64" ? Float64Array.from(d) : dt === "int32" || dt === "int64" ? Int32Array.from(d)
          : dt === "uint8" || dt === "bool" ? Uint8Array.from(d) : Float32Array.from(d);
      }
      shapes.set(arr, shape);
      return arr;
    }
    const o: any = {};
    for (const [k, x] of Object.entries(v)) o[k] = resolve(x, dir);
    return o;
  }
  return v;
}

/** 楽譜 1 つぶんの正解（case.json）。phrases[i] に句ごとの段の値が入る */
export function loadCase(name: string): any {
  const dir = join(goldenDir(), name);
  return resolve(JSON.parse(readFileSync(join(dir, "case.json"), "utf-8")), dir);
}

// ケースごとのボイスバンク（tools/golden/dump_golden.py のケースの "vb"。無ければ日本語）
const CASE_VB: Record<string, string> = { korean: "leapsinger_ko_mechol_sodasoo" };
const vbKey = (name: string): string => CASE_VB[name] ?? "leapsinger_ja_3singer";

/** そのケースのボイスバンクが手元にあるか（韓国語のボイスバンクは同梱していない） */
export const hasVb = (name: string): boolean => existsSync(join(repoRoot(), `voicebanks/${vbKey(name)}/web.json`));

/** 照らし合わせるケース。ボイスバンクが手元に無いケースは、その旨を出して飛ばす */
export function cases(): string[] {
  return ["sakura", "mixed", "params", "dips", "korean"].filter((name) => {
    if (hasVb(name)) return true;
    console.log(`（${name} は飛ばす: ボイスバンク ${vbKey(name)} が手元に無い）`);
    return false;
  });
}

const vbCache = new Map<string, any>();
/** そのケースを書き出したボイスバンクの web.json */
export function vbFor(name: string): any {
  const key = vbKey(name);
  if (!vbCache.has(key)) vbCache.set(key, JSON.parse(readRepo(`voicebanks/${key}/web.json`)));
  return vbCache.get(key);
}

let failures = 0;
/** 2 つの数の並びを比べ、最大の差を出す。|差| ≤ atol + rtol·|正解| をすべて満たせば合格 */
export function close(label: string, got: ArrayLike<number>, want: ArrayLike<number>, atol = 1e-6, rtol = 0): boolean {
  if (got.length !== want.length) {
    console.log(`NG  ${label}: 長さが違う（${got.length} と正解 ${want.length}）`);
    failures++;
    return false;
  }
  let maxd = 0;
  let at = -1;
  let bad = 0;
  for (let i = 0; i < got.length; i++) {
    const d = Math.abs(Number(got[i]) - Number(want[i]));
    if (!(d <= atol + rtol * Math.abs(Number(want[i])))) bad++;
    if (d > maxd || Number.isNaN(d)) { maxd = d; at = i; }
  }
  const ok = bad === 0;
  if (!ok) failures++;
  console.log(`${ok ? "OK" : "NG"}  ${label}: 最大の差 ${maxd.toExponential(2)}（${at} 番目）${ok ? "" : `、外れ ${bad} 個`}`);
  return ok;
}

/** 値がそのまま等しいか（文字列・整数の並び・オブジェクト） */
export function same(label: string, got: unknown, want: unknown): boolean {
  const g = JSON.stringify(got, (_, x) => (ArrayBuffer.isView(x) ? Array.from(x as unknown as number[]) : x));
  const w = JSON.stringify(want, (_, x) => (ArrayBuffer.isView(x) ? Array.from(x as unknown as number[]) : x));
  const ok = g === w;
  if (!ok) failures++;
  console.log(`${ok ? "OK" : "NG"}  ${label}${ok ? "" : `\n    得た値: ${g.slice(0, 400)}\n    正解:   ${w.slice(0, 400)}`}`);
  return ok;
}

/** 最後に呼ぶ: 外れがあれば終了コード 1 */
export function done(): void {
  console.log(failures ? `\n外れ ${failures} 件` : "\nすべて一致");
  if (failures) process.exitCode = 1;
}
